// FindAndApplyLocalThresholding(Im, dw = w, dh = 32, w, h) on the GPU.
//
// CPU algorithm (IPAlgorithms.cpp), for every strip of dh rows (the last one has h % dh rows):
//   histogram of the non zero values, min and max of the non zero values, mid = (min + max) / 2
//   li = leftmost mode in [min, mid)   (li = min if the range is empty),   ri = leftmost mode in [mid, max]
//   thr = (lmax < rmax) ? li : ri,  val = the matching count
//   for i in (li, ri): if hist[i] < val -> thr = i, val = hist[i]   (so thr is the leftmost minimum if it is below the initial val)
//   every value < thr in the strip is set to 0
// GPU: one block per strip. Atomic histogram in global memory, block reductions with the same tie breaking (smaller index wins).
#include "kernels.h"
#include <limits.h>

namespace gpu_transform
{
namespace kernels
{

namespace
{

constexpr int BLK = 256;
constexpr int STRIP = 32;

struct Best { unsigned cnt; int idx; };

// "better" for the maximum search: higher count, on a tie the smaller index
__device__ __forceinline__ bool better_max(Best a, Best b) { return (a.cnt > b.cnt) || (a.cnt == b.cnt && a.idx < b.idx); }
// "better" for the minimum search: lower count, on a tie the smaller index
__device__ __forceinline__ bool better_min(Best a, Best b) { return (a.cnt < b.cnt) || (a.cnt == b.cnt && a.idx < b.idx); }

template <bool MAXIMUM>
__device__ Best block_best(Best mine, Best* sh)
{
	sh[threadIdx.x] = mine;
	__syncthreads();
	for (int s = BLK / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
		{
			Best o = sh[threadIdx.x + s];
			if (MAXIMUM ? better_max(o, sh[threadIdx.x]) : better_min(o, sh[threadIdx.x])) sh[threadIdx.x] = o;
		}
		__syncthreads();
	}
	Best r = sh[0];
	__syncthreads();
	return r;
}

__global__ void k_find_thr(const uint16_t* __restrict__ im, int w, int h, unsigned* hist, int* thr_out)
{
	__shared__ Best sh[BLK];
	__shared__ int s_min[BLK], s_max[BLK];
	__shared__ int s_thr, s_val, s_li, s_ri, s_lmax, s_rmax, s_min_all, s_max_all;

	const int tile = blockIdx.x;
	const int y0 = tile * STRIP;
	const int y1 = min(h, y0 + STRIP);
	unsigned* hs = hist + (size_t)tile * MAX_EDGE_STR;

	// pass 1: min / max of the non zero values and the histogram
	int tmin = INT_MAX, tmax = 0;
	const int n = (y1 - y0) * w;
	const uint16_t* p = im + (size_t)y0 * w;
	for (int i = threadIdx.x; i < n; i += BLK)
	{
		int v = p[i];
		if (v == 0) continue;
		if (v > tmax) tmax = v;
		if (v < tmin) tmin = v;
		atomicAdd(&hs[v], 1u);
	}
	s_min[threadIdx.x] = tmin; s_max[threadIdx.x] = tmax;
	__syncthreads();
	for (int s = BLK / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
		{
			s_min[threadIdx.x] = min(s_min[threadIdx.x], s_min[threadIdx.x + s]);
			s_max[threadIdx.x] = max(s_max[threadIdx.x], s_max[threadIdx.x + s]);
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) { s_min_all = s_min[0]; s_max_all = s_max[0]; }
	__syncthreads();
	const int mn = s_min_all, mx = s_max_all;

	if (mx == 0)   // no non zero value in the strip: nothing to do
	{
		if (threadIdx.x == 0) thr_out[tile] = 0;
		return;
	}
	const int mid = (mn + mx) / 2;

	// the atomics of the other threads of this block must be visible: read the histogram through L2 (__ldcg)
	__threadfence();
	__syncthreads();

	// li: leftmost mode in [mn, mid)  (li = mn when the range is empty)
	Best bl = { 0u, INT_MAX };
	for (int i = mn + threadIdx.x; i < mid; i += BLK) { Best c = { __ldcg(&hs[i]), i }; if (better_max(c, bl)) bl = c; }
	Best L = block_best<true>(bl, sh);
	if (threadIdx.x == 0)
	{
		if (mid > mn) { s_li = L.idx; s_lmax = (int)L.cnt; }
		else { s_li = mn; s_lmax = (int)__ldcg(&hs[mn]); }
	}
	// ri: leftmost mode in [mid, mx]
	Best br = { 0u, INT_MAX };
	for (int i = mid + threadIdx.x; i <= mx; i += BLK) { Best c = { __ldcg(&hs[i]), i }; if (better_max(c, br)) br = c; }
	Best R = block_best<true>(br, sh);
	if (threadIdx.x == 0)
	{
		s_ri = R.idx; s_rmax = (int)R.cnt;
		if (s_lmax < s_rmax) { s_thr = s_li; s_val = s_lmax; }
		else { s_thr = s_ri; s_val = s_rmax; }
	}
	__syncthreads();

	// leftmost minimum in (li, ri); it replaces thr only if it is lower than val
	Best bm = { UINT_MAX, INT_MAX };
	for (int i = s_li + 1 + threadIdx.x; i < s_ri; i += BLK) { Best c = { __ldcg(&hs[i]), i }; if (better_min(c, bm)) bm = c; }
	Best M = block_best<false>(bm, sh);
	if (threadIdx.x == 0)
	{
		int thr = s_thr;
		if (M.idx != INT_MAX && (int)M.cnt < s_val) thr = M.idx;
		thr_out[tile] = thr;
	}
}

__global__ void k_apply_thr(uint16_t* __restrict__ im, int w, int h, const int* __restrict__ thr)
{
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= (size_t)w * h) return;
	int y = (int)(i / w);
	if ((int)im[i] < thr[y / STRIP]) im[i] = 0;
}

}

cudaError_t LocalThreshold(uint16_t* im, int w, int h, unsigned* hist, int* thr, cudaStream_t st)
{
	if (w <= 0 || h <= 0) return cudaSuccess;
	const int ntiles = (h + STRIP - 1) / STRIP;
	cudaError_t e = cudaMemsetAsync(hist, 0, LocalThresholdHistBytes(h), st);
	if (e != cudaSuccess) return e;
	k_find_thr<<<ntiles, BLK, 0, st>>>(im, w, h, hist, thr);
	e = cudaGetLastError();
	if (e != cudaSuccess) return e;
	const size_t n = (size_t)w * h;
	k_apply_thr<<<(unsigned)((n + 255) / 256), 256, 0, st>>>(im, w, h, thr);
	return cudaGetLastError();
}

}
}
