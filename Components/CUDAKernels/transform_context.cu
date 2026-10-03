#include "transform_context.h"
#include "kernels/kernels.h"
#include <vector>

#define GT_RETURN_IF(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) return e_; } while (0)

namespace gpu_transform
{

Context::~Context()
{
	Free();
	if (st_) cudaStreamDestroy(st_);
}

void Context::Free()
{
	if (block_)
	{
		cudaStreamSynchronize(st_);
		cudaFree(block_);
	}
	block_ = nullptr;
	cap_pixels_ = 0; cap_hist_bytes_ = 0;
	bgr = y = u = v = gy = gu = gv = orband = ff = ne = he = nullptr;
	moeY = moeU = moeV = noeY = noeU = noeV = r1 = r2 = cm[0] = cm[1] = ess = ecp = nullptr;
	dmax_ = nullptr; hist_ = nullptr; thr_ = nullptr;
}

cudaError_t Context::Init()
{
	if (st_) return cudaSuccess;
	return cudaStreamCreateWithFlags(&st_, cudaStreamNonBlocking);
}

cudaError_t Context::Reserve(int w, int h)
{
	if (w <= 0 || h <= 0) return cudaErrorInvalidValue;
	const size_t n = (size_t)w * h;
	const size_t hist_bytes = kernels::LocalThresholdHistBytes(h);
	if (block_ && n <= cap_pixels_ && hist_bytes <= cap_hist_bytes_)
	{
		w_ = w; h_ = h;
		return cudaSuccess;
	}

	Free();
	w_ = 0; h_ = 0;

	const size_t align = 256;
	auto up = [&](size_t x) { return (x + align - 1) / align * align; };
	const size_t b8 = up(n), b16 = up(n * 2);
	const size_t total = up(n * 3) + 10 * b8 + 12 * b16 + up(kernels::MAX_BANDS * sizeof(unsigned)) + up(hist_bytes) + up(kernels::LocalThresholdThrBytes(h));

	char* p = nullptr;
	cudaError_t e = cudaMalloc((void**)&p, total);
	if (e != cudaSuccess) return e;
	block_ = p;
	cap_pixels_ = n; cap_hist_bytes_ = hist_bytes;
	w_ = w; h_ = h;

	auto take8 = [&]() { uint8_t* r = (uint8_t*)p; p += b8; return r; };
	auto take16 = [&]() { uint16_t* r = (uint16_t*)p; p += b16; return r; };
	bgr = (uint8_t*)p; p += up(n * 3);
	y = take8(); u = take8(); v = take8(); gy = take8(); gu = take8(); gv = take8(); orband = take8(); ff = take8(); ne = take8(); he = take8();
	moeY = take16(); moeU = take16(); moeV = take16(); noeY = take16(); noeU = take16(); noeV = take16();
	r1 = take16(); r2 = take16(); cm[0] = take16(); cm[1] = take16(); ess = take16(); ecp = take16();
	dmax_ = (unsigned*)p; p += up(kernels::MAX_BANDS * sizeof(unsigned));
	hist_ = (unsigned*)p; p += up(hist_bytes);
	thr_ = (int*)p;
	return cudaSuccess;
}

bool Context::BandsValid(int h, int N, const int* LB, const int* LE)
{
	if (N < 1 || N > kernels::MAX_BANDS || h < 1 || !LB || !LE) return false;
	long long sum = 0;
	for (int k = 0; k < N; k++)
	{
		if (LB[k] < 0 || LE[k] < LB[k] || LE[k] >= h) return false;
		if (k > 0 && LB[k] <= LE[k - 1]) return false;
		sum += LE[k] - LB[k] + 1;
	}
	return sum <= h;
}

cudaError_t Context::ComputeYuv()
{
	return kernels::BgrToYuv(bgr, y, u, v, w_ * h_, st_);
}

cudaError_t Context::ComputeLocalThreshold(uint16_t* im, int w, int h)
{
	return kernels::LocalThreshold(im, w, h, hist_, thr_, st_);
}

cudaError_t Context::ComputeCmoe(int variant, int eh, int N, const int* row0, const int* rows, double mthr)
{
	const int w = w_;
	const size_t n = (size_t)w * eh;
	uint16_t* c = cm[variant];
	GT_RETURN_IF(cudaMemsetAsync(c, 0, n * 2, st_));                                  // EasyBorderClear
	GT_RETURN_IF(kernels::WeightedSum(moeY, moeU, moeV, c, w, eh, variant, st_));
	GT_RETURN_IF(ComputeLocalThreshold(c, w, eh));                // FindAndApplyLocalThresholding(ImCMOE, w, 32, w, h)
	GT_RETURN_IF(cudaMemsetAsync(ess, 0, n * 2, st_));
	GT_RETURN_IF(cudaMemsetAsync(ecp, 0, n * 2, st_));
	GT_RETURN_IF(kernels::Ess(c, ess, w, eh, st_));                                   // the 2 pixel border stays 0 (BorderClear)
	GT_RETURN_IF(kernels::Ecp(ess, ecp, w, eh, st_));
	GT_RETURN_IF(kernels::CmoeCombine(c, ess, ecp, w, eh, st_));
	for (int k = 0; k < N; k++) GT_RETURN_IF(kernels::ModerateThreshold(c, w, row0[k], rows[k], mthr, dmax_ + k, st_));
	return cudaSuccess;
}

cudaError_t Context::ComputeFF(int N, const int* LB, const int* LE, double mthr)
{
	if (!BandsValid(h_, N, LB, LE)) return cudaErrorInvalidValue;
	const int w = w_;
	std::vector<int> row0(N), rows(N);
	int eh = 0;                                                                       // height of the gathered image
	for (int k = 0; k < N; k++) { rows[k] = LE[k] - LB[k] + 1; row0[k] = eh; eh += rows[k]; }

	GT_RETURN_IF(kernels::GatherBands(y, gy, w, N, LB, LE, st_));
	GT_RETURN_IF(kernels::GatherBands(u, gu, w, N, LB, LE, st_));
	GT_RETURN_IF(kernels::GatherBands(v, gv, w, N, LB, LE, st_));

	const size_t bytes16 = (size_t)w * eh * 2;
	GT_RETURN_IF(cudaMemsetAsync(moeY, 0, bytes16, st_));
	GT_RETURN_IF(cudaMemsetAsync(moeU, 0, bytes16, st_));
	GT_RETURN_IF(cudaMemsetAsync(moeV, 0, bytes16, st_));
	GT_RETURN_IF(kernels::SobelM(gy, moeY, w, eh, st_));
	GT_RETURN_IF(kernels::SobelM(gu, moeU, w, eh, st_));
	GT_RETURN_IF(kernels::SobelM(gv, moeV, w, eh, st_));

	GT_RETURN_IF(ComputeCmoe(0, eh, N, row0.data(), rows.data(), mthr));
	GT_RETURN_IF(ComputeCmoe(1, eh, N, row0.data(), rows.data(), mthr));
	GT_RETURN_IF(kernels::OrAll(cm[1], cm[0], orband, w * eh, st_));
	return kernels::ScatterBands(orband, ff, w, h_, N, LB, LE, st_);
}

// GetImNE (SobelN) and GetImHE (SobelH) differ only in the edge operator and the output image
static cudaError_t ComputeEdges(Context& c, bool horizontal, uint8_t* out)
{
	const int w = c.Width(), h = c.Height();
	const size_t n = (size_t)w * h;
	cudaStream_t st = c.Stream();
	if (horizontal)
	{
		GT_RETURN_IF(kernels::SobelH(c.y, c.noeY, w, h, st));
		GT_RETURN_IF(kernels::SobelH(c.u, c.noeU, w, h, st));
		GT_RETURN_IF(kernels::SobelH(c.v, c.noeV, w, h, st));
	}
	else
	{
		GT_RETURN_IF(kernels::SobelN(c.y, c.noeY, w, h, st));
		GT_RETURN_IF(kernels::SobelN(c.u, c.noeU, w, h, st));
		GT_RETURN_IF(kernels::SobelN(c.v, c.noeV, w, h, st));
	}
	GT_RETURN_IF(cudaMemsetAsync(c.r1, 0, n * 2, st));
	GT_RETURN_IF(cudaMemsetAsync(c.r2, 0, n * 2, st));
	GT_RETURN_IF(cudaMemsetAsync(out, 0, n, st));
	GT_RETURN_IF(kernels::WeightedSum(c.noeY, c.noeU, c.noeV, c.r1, w, h, 0, st));
	GT_RETURN_IF(kernels::WeightedSum(c.noeY, c.noeU, c.noeV, c.r2, w, h, 1, st));
	return cudaSuccess;
}

cudaError_t Context::ComputeNE(double mnthr)
{
	GT_RETURN_IF(ComputeEdges(*this, false, ne));
	GT_RETURN_IF(kernels::ModerateThreshold(r1, w_, 0, h_, mnthr, dmax_ + 0, st_));
	GT_RETURN_IF(kernels::ModerateThreshold(r2, w_, 0, h_, mnthr, dmax_ + 1, st_));
	return kernels::OrInterior(r1, r2, ne, w_, h_, st_);
}

cudaError_t Context::ComputeHE(double mnthr)
{
	GT_RETURN_IF(ComputeEdges(*this, true, he));
	GT_RETURN_IF(kernels::ModerateThreshold(r1, w_, 0, h_, mnthr, dmax_ + 0, st_));
	GT_RETURN_IF(kernels::ModerateThreshold(r2, w_, 0, h_, mnthr, dmax_ + 1, st_));
	return kernels::OrInterior(r1, r2, he, w_, h_, st_);
}

cudaError_t Context::Transform(int N, const int* LB, const int* LE, double mthr, double mnthr)
{
	if (!BandsValid(h_, N, LB, LE)) return cudaErrorInvalidValue;
	GT_RETURN_IF(ComputeYuv());
	GT_RETURN_IF(ComputeFF(N, LB, LE, mthr));
	GT_RETURN_IF(ComputeNE(mnthr));
	GT_RETURN_IF(ComputeHE(mnthr));
	return kernels::CombineImages(ne, he, w_ * h_, st_);
}

}
