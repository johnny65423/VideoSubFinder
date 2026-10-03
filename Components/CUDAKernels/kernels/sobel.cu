// ImprovedSobelMEdge, FastImprovedSobelNEdge, FastImprovedSobelHEdge (IPAlgorithms.cpp): 3x3 neighbourhoods, 16 bit arithmetic as in the CPU code.
// One thread per pixel, no shared memory yet (M4 tunes this).
#include "kernels.h"

namespace gpu_transform
{
namespace kernels
{

static __global__ void k_sobel_m(const uint8_t* __restrict__ ImIn, uint16_t* __restrict__ ImMOE, int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	if (x >= w - 1 || y >= h - 1) return;
	const uint8_t* pIm = ImIn + (size_t)y * w + x;
	short val, val1, val2, val3, val4, max;

	val1 = (short)(*(pIm - w - 1)) - (short)(*(pIm + w + 1));
	val2 = (short)(*(pIm - w + 1)) - (short)(*(pIm + w - 1));
	val3 = (short)(*(pIm - w)) - (short)(*(pIm + w));
	val4 = (short)(*(pIm - 1)) - (short)(*(pIm + 1));

	val = 3 * (val1 + val2) + 10 * val3;
	if (val < 0) max = -val; else max = val;

	val = 3 * (val1 - val2) + 10 * val4;
	if (val < 0) val = -val;
	if (max < val) max = val;

	val = 3 * (val3 + val4) + 10 * val1;
	if (val < 0) val = -val;
	if (max < val) max = val;

	val = 3 * (val3 - val4) + 10 * val2;
	if (val < 0) val = -val;
	if (max < val) max = val;

	ImMOE[(size_t)y * w + x] = (uint16_t)max;
}

static __global__ void k_sobel_n(const uint8_t* __restrict__ ImIn, uint16_t* __restrict__ ImNOE, int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	if (x >= w - 1 || y >= h - 1) return;
	const uint8_t* pIm = ImIn + (size_t)y * w + x;
	short val, val1, val2;

	val1 = (short)(*(pIm - w));
	val2 = (short)(*(pIm - w - 1));
	val1 += (short)(*(pIm - 1)) - (short)(*(pIm + 1));
	val1 -= (short)(*(pIm + w));
	val2 -= (short)(*(pIm + w + 1));
	val = 3 * val1 + 10 * val2;
	if (val < 0) val = -val;
	ImNOE[(size_t)y * w + x] = (uint16_t)val;
}

static __global__ void k_sobel_h(const uint8_t* __restrict__ ImIn, uint16_t* __restrict__ ImHOE, int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
	if (x >= w - 1 || y >= h - 1) return;
	const uint8_t* pIm = ImIn + (size_t)y * w + x;
	short val, val1, val2;

	val1 = (short)(*(pIm - w - 1)) + (short)(*(pIm - w + 1));
	val2 = (short)(*(pIm - w));
	val1 -= (short)(*(pIm + w - 1)) + (short)(*(pIm + w + 1));
	val2 -= (short)(*(pIm + w));
	val = 3 * val1 + 10 * val2;
	if (val < 0) val = -val;
	ImHOE[(size_t)y * w + x] = (uint16_t)val;
}

static dim3 Grid(int w, int h) { return dim3((unsigned)max(1, (w - 1 + 31) / 32), (unsigned)max(1, (h - 1 + 7) / 8)); }

cudaError_t SobelM(const uint8_t* in, uint16_t* out, int w, int h, cudaStream_t st)
{
	k_sobel_m<<<Grid(w, h), dim3(32, 8), 0, st>>>(in, out, w, h);
	return cudaGetLastError();
}

cudaError_t SobelN(const uint8_t* in, uint16_t* out, int w, int h, cudaStream_t st)
{
	k_sobel_n<<<Grid(w, h), dim3(32, 8), 0, st>>>(in, out, w, h);
	return cudaGetLastError();
}

cudaError_t SobelH(const uint8_t* in, uint16_t* out, int w, int h, cudaStream_t st)
{
	k_sobel_h<<<Grid(w, h), dim3(32, 8), 0, st>>>(in, out, w, h);
	return cudaGetLastError();
}

}
}
