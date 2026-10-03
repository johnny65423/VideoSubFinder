#include "gpu_transform_test.h"
#include "transform_context.h"
#include "kernels/kernels.h"

#define GT_RETURN_IF(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) return (int)e_; } while (0)

namespace gpu_transform
{
namespace testing
{

namespace
{

// one context per thread, as the production code will do
Context& Ctx()
{
	thread_local Context ctx;
	return ctx;
}

int Prepare(Context& c, int w, int h)
{
	GT_RETURN_IF(c.Init());
	return (int)c.Reserve(w, h);
}

template <class T>
cudaError_t Up(T* dev, const T* host, size_t count, cudaStream_t st) { return cudaMemcpyAsync(dev, host, count * sizeof(T), cudaMemcpyHostToDevice, st); }

template <class T>
cudaError_t Down(T* host, const T* dev, size_t count, cudaStream_t st) { return cudaMemcpyAsync(host, dev, count * sizeof(T), cudaMemcpyDeviceToHost, st); }

int Finish(Context& c, cudaError_t e)
{
	cudaError_t s = cudaStreamSynchronize(c.Stream());
	return (int)(e != cudaSuccess ? e : s);
}

}

int BgrToYuv(const uint8_t* bgr, int w, int h, uint8_t* y, uint8_t* u, uint8_t* v)
{
	Context& c = Ctx();
	GT_RETURN_IF((cudaError_t)Prepare(c, w, h));
	const size_t n = (size_t)w * h;
	cudaError_t e = Up(c.bgr, bgr, n * 3, c.Stream());
	if (e == cudaSuccess) e = c.ComputeYuv();
	if (e == cudaSuccess) e = Down(y, c.y, n, c.Stream());
	if (e == cudaSuccess) e = Down(u, c.u, n, c.Stream());
	if (e == cudaSuccess) e = Down(v, c.v, n, c.Stream());
	return Finish(c, e);
}

int Sobel(char kind, const uint8_t* in, int w, int h, uint16_t* out)
{
	Context& c = Ctx();
	GT_RETURN_IF((cudaError_t)Prepare(c, w, h));
	const size_t n = (size_t)w * h;
	cudaError_t e = Up(c.y, in, n, c.Stream());
	if (e == cudaSuccess) e = Up(c.moeY, out, n, c.Stream());
	if (e == cudaSuccess)
	{
		if (kind == 'M') e = kernels::SobelM(c.y, c.moeY, w, h, c.Stream());
		else if (kind == 'N') e = kernels::SobelN(c.y, c.moeY, w, h, c.Stream());
		else if (kind == 'H') e = kernels::SobelH(c.y, c.moeY, w, h, c.Stream());
		else e = cudaErrorInvalidValue;
	}
	if (e == cudaSuccess) e = Down(out, c.moeY, n, c.Stream());
	return Finish(c, e);
}

int LocalThreshold(uint16_t* im, int w, int h)
{
	Context& c = Ctx();
	GT_RETURN_IF((cudaError_t)Prepare(c, w, h));
	const size_t n = (size_t)w * h;
	cudaError_t e = Up(c.cm[0], im, n, c.Stream());
	if (e == cudaSuccess) e = c.ComputeLocalThreshold(c.cm[0], w, h);
	if (e == cudaSuccess) e = Down(im, c.cm[0], n, c.Stream());
	return Finish(c, e);
}

int Convolutions(const uint16_t* in, int w, int h, uint16_t* ess, uint16_t* ecp)
{
	Context& c = Ctx();
	GT_RETURN_IF((cudaError_t)Prepare(c, w, h));
	const size_t n = (size_t)w * h;
	cudaError_t e = Up(c.cm[0], in, n, c.Stream());
	if (e == cudaSuccess) e = Up(c.ess, ess, n, c.Stream());
	if (e == cudaSuccess) e = Up(c.ecp, ecp, n, c.Stream());
	if (e == cudaSuccess) e = kernels::Ess(c.cm[0], c.ess, w, h, c.Stream());
	if (e == cudaSuccess) e = kernels::Ecp(c.cm[0], c.ecp, w, h, c.Stream());
	if (e == cudaSuccess) e = Down(ess, c.ess, n, c.Stream());
	if (e == cudaSuccess) e = Down(ecp, c.ecp, n, c.Stream());
	return Finish(c, e);
}

static int Edges(bool horizontal, const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h, double mnthr, uint8_t* out)
{
	Context& c = Ctx();
	GT_RETURN_IF((cudaError_t)Prepare(c, w, h));
	const size_t n = (size_t)w * h;
	cudaError_t e = Up(c.y, y, n, c.Stream());
	if (e == cudaSuccess) e = Up(c.u, u, n, c.Stream());
	if (e == cudaSuccess) e = Up(c.v, v, n, c.Stream());
	if (e == cudaSuccess) e = horizontal ? c.ComputeHE(mnthr) : c.ComputeNE(mnthr);
	if (e == cudaSuccess) e = Down(out, horizontal ? c.he : c.ne, n, c.Stream());
	return Finish(c, e);
}

int GetImNE(const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h, double mnthr, uint8_t* out) { return Edges(false, y, u, v, w, h, mnthr, out); }
int GetImHE(const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h, double mnthr, uint8_t* out) { return Edges(true, y, u, v, w, h, mnthr, out); }

int GetImCMOE(int variant, const uint16_t* moeY, const uint16_t* moeU, const uint16_t* moeV, int w, int h, int N, const int* row0, const int* rows, double mthr, uint16_t* out)
{
	if (variant < 0 || variant > 1) return (int)cudaErrorInvalidValue;
	Context& c = Ctx();
	GT_RETURN_IF((cudaError_t)Prepare(c, w, h));
	const size_t n = (size_t)w * h;
	cudaError_t e = Up(c.moeY, moeY, n, c.Stream());
	if (e == cudaSuccess) e = Up(c.moeU, moeU, n, c.Stream());
	if (e == cudaSuccess) e = Up(c.moeV, moeV, n, c.Stream());
	if (e == cudaSuccess) e = c.ComputeCmoe(variant, h, N, row0, rows, mthr);
	if (e == cudaSuccess) e = Down(out, c.cm[variant], n, c.Stream());
	return Finish(c, e);
}

int GetImFF(const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h, int N, const int* LB, const int* LE, double mthr, uint8_t* ff)
{
	Context& c = Ctx();
	GT_RETURN_IF((cudaError_t)Prepare(c, w, h));
	const size_t n = (size_t)w * h;
	cudaError_t e = Up(c.y, y, n, c.Stream());
	if (e == cudaSuccess) e = Up(c.u, u, n, c.Stream());
	if (e == cudaSuccess) e = Up(c.v, v, n, c.Stream());
	if (e == cudaSuccess) e = c.ComputeFF(N, LB, LE, mthr);
	if (e == cudaSuccess) e = Down(ff, c.ff, n, c.Stream());
	return Finish(c, e);
}

int Transform(const uint8_t* bgr, int w, int h, int N, const int* LB, const int* LE, double mthr, double mnthr, uint8_t* ff, uint8_t* ne, uint8_t* y)
{
	Context& c = Ctx();
	GT_RETURN_IF((cudaError_t)Prepare(c, w, h));
	const size_t n = (size_t)w * h;
	cudaError_t e = Up(c.bgr, bgr, n * 3, c.Stream());
	if (e == cudaSuccess) e = c.Transform(N, LB, LE, mthr, mnthr);
	if (e == cudaSuccess) e = Down(ff, c.ff, n, c.Stream());
	if (e == cudaSuccess) e = Down(ne, c.ne, n, c.Stream());
	if (e == cudaSuccess) e = Down(y, c.y, n, c.Stream());
	return Finish(c, e);
}

}
}
