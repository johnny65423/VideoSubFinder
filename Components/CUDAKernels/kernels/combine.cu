// Weighted sums, ApplyModerateThreshold and the merging steps of GetImFF / GetImNE / GetImHE / GetTransformedImage (IPAlgorithms.cpp).
#include "kernels.h"

namespace gpu_transform
{
namespace kernels
{

namespace
{

// ImRES = Y + U + V (mode 0) or Y + (U + V) * 5 (mode 1), interior only
__global__ void k_sum(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, const uint16_t* __restrict__ c, uint16_t* __restrict__ out, int w, int h, int mode)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	if (x >= w - 1 || y >= h - 1) return;
	size_t i = (size_t)y * w + x;
	out[i] = (uint16_t)(mode == 0 ? (a[i] + b[i] + c[i]) : (a[i] + (b[i] + c[i]) * 5));
}

// ApplyModerateThreshold, part 1: maximum of the region
__global__ void k_region_max(const uint16_t* __restrict__ im, size_t count, unsigned* dmax)
{
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	unsigned v = (i < count) ? im[i] : 0u;
	for (int o = 16; o > 0; o >>= 1) v = max(v, (unsigned)__shfl_down_sync(0xffffffffu, v, o));
	if ((threadIdx.x & 31) == 0 && v) atomicMax(dmax, v);
}

// ApplyModerateThreshold, part 2: thr = (T)((double)mx * mthr); value < thr -> 0, otherwise 255 (also for zero values if thr == 0!)
__global__ void k_region_thr(uint16_t* __restrict__ im, size_t count, const unsigned* __restrict__ dmax, double mthr)
{
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count) return;
	uint16_t mx = (uint16_t)(*dmax);
	uint16_t thr = (uint16_t)((double)mx * mthr);
	im[i] = (im[i] < thr) ? (uint16_t)0 : (uint16_t)255;
}

__global__ void k_cmoe_combine(uint16_t* __restrict__ cm, const uint16_t* __restrict__ r2, const uint16_t* __restrict__ r3, int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= w || y >= h) return;
	size_t i = (size_t)y * w + x;
	if (x < 2 || y < 2 || x >= w - 2 || y >= h - 2) cm[i] = 0;
	else cm[i] = (uint16_t)(((int)r2[i] + (int)r3[i]) / 2);
}

__global__ void k_or_interior(const uint16_t* __restrict__ r1, const uint16_t* __restrict__ r2, uint8_t* __restrict__ out, int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	if (x >= w - 1 || y >= h - 1) return;
	size_t i = (size_t)y * w + x;
	out[i] = (r1[i] || r2[i]) ? 255 : 0;
}

__global__ void k_or_all(const uint16_t* __restrict__ a, const uint16_t* __restrict__ b, uint8_t* __restrict__ out, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) out[i] = (a[i] || b[i]) ? 255 : 0;
}

__global__ void k_combine_images(uint8_t* __restrict__ ne, const uint8_t* __restrict__ he, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && ne[i] == 0 && he[i] != 0) ne[i] = 255;
}

// grid for an image without a border of the given width on every side
dim3 Grid(int w, int h, int border) { return dim3((unsigned)max(1, (w - 2 * border + 31) / 32), (unsigned)max(1, (h - 2 * border + 7) / 8)); }

}

cudaError_t WeightedSum(const uint16_t* a, const uint16_t* b, const uint16_t* c, uint16_t* out, int w, int h, int mode, cudaStream_t st)
{
	k_sum<<<Grid(w, h, 1), dim3(32, 8), 0, st>>>(a, b, c, out, w, h, mode);
	return cudaGetLastError();
}

cudaError_t ModerateThreshold(uint16_t* im, int w, int row0, int rows, double mthr, unsigned* dmax, cudaStream_t st)
{
	const size_t count = (size_t)rows * w;
	if (count == 0) return cudaSuccess;
	cudaError_t e = cudaMemsetAsync(dmax, 0, sizeof(unsigned), st);
	if (e != cudaSuccess) return e;
	uint16_t* region = im + (size_t)row0 * w;
	k_region_max<<<(unsigned)((count + 255) / 256), 256, 0, st>>>(region, count, dmax);
	e = cudaGetLastError();
	if (e != cudaSuccess) return e;
	k_region_thr<<<(unsigned)((count + 255) / 256), 256, 0, st>>>(region, count, dmax, mthr);
	return cudaGetLastError();
}

cudaError_t CmoeCombine(uint16_t* cm, const uint16_t* r2, const uint16_t* r3, int w, int h, cudaStream_t st)
{
	k_cmoe_combine<<<dim3((unsigned)((w + 31) / 32), (unsigned)((h + 7) / 8)), dim3(32, 8), 0, st>>>(cm, r2, r3, w, h);
	return cudaGetLastError();
}

cudaError_t OrInterior(const uint16_t* a, const uint16_t* b, uint8_t* out, int w, int h, cudaStream_t st)
{
	k_or_interior<<<Grid(w, h, 1), dim3(32, 8), 0, st>>>(a, b, out, w, h);
	return cudaGetLastError();
}

cudaError_t OrAll(const uint16_t* a, const uint16_t* b, uint8_t* out, int n, cudaStream_t st)
{
	if (n <= 0) return cudaSuccess;
	k_or_all<<<(n + 255) / 256, 256, 0, st>>>(a, b, out, n);
	return cudaGetLastError();
}

cudaError_t CombineImages(uint8_t* ne, const uint8_t* he, int n, cudaStream_t st)
{
	if (n <= 0) return cudaSuccess;
	k_combine_images<<<(n + 255) / 256, 256, 0, st>>>(ne, he, n);
	return cudaGetLastError();
}

}
}
