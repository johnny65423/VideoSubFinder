// AplyESS and AplyECP (IPAlgorithms.cpp): 5x5 convolutions on 16 bit images, 32 bit integer arithmetic.
#include "kernels.h"

namespace gpu_transform
{
namespace kernels
{

static __global__ void k_ess(const uint16_t* __restrict__ ImIn, uint16_t* __restrict__ ImOut, int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + 2;
	int y = blockIdx.y * blockDim.y + threadIdx.y + 2;
	if (x >= w - 2 || y >= h - 2) return;
	int i = w * y + x;
	int val = 2 * ((int)ImIn[i - w * 2 - 2] + (int)ImIn[i - w * 2 + 2] + (int)ImIn[i + w * 2 - 2] + (int)ImIn[i + w * 2 + 2]) +
		+4 * ((int)ImIn[i - w * 2 - 1] + (int)ImIn[i - w * 2 + 1] + (int)ImIn[i - w - 2] + (int)ImIn[i - w + 2] + (int)ImIn[i + w - 2] + (int)ImIn[i + w + 2] + (int)ImIn[i + w * 2 - 1] + (int)ImIn[i + w * 2 + 1]) +
		+5 * ((int)ImIn[i - w * 2] + (int)ImIn[i - 2] + (int)ImIn[i + 2] + (int)ImIn[i + w * 2]) +
		+10 * ((int)ImIn[i - w - 1] + (int)ImIn[i - w + 1] + (int)ImIn[i + w - 1] + (int)ImIn[i + w + 1]) +
		+20 * ((int)ImIn[i - w] + (int)ImIn[i - 1] + (int)ImIn[i + 1] + (int)ImIn[i + w]) +
		+40 * (int)ImIn[i];
	ImOut[i] = (uint16_t)(val / 220);
}

static __global__ void k_ecp(const uint16_t* __restrict__ ImIn, uint16_t* __restrict__ ImOut, int w, int h)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x + 2;
	int y = blockIdx.y * blockDim.y + threadIdx.y + 2;
	if (x >= w - 2 || y >= h - 2) return;
	int i = w * y + x;
	if (ImIn[i] == 0) { ImOut[i] = 0; return; }   // the CPU code skips the convolution for zero pixels
	int ii = i - ((w + 1) << 1);
	int val = 8 * (int)ImIn[ii] + 5 * (int)ImIn[ii + 1] + 4 * (int)ImIn[ii + 2] + 5 * (int)ImIn[ii + 3] + 8 * (int)ImIn[ii + 4];
	ii += w;
	val += 5 * (int)ImIn[ii] + 2 * (int)ImIn[ii + 1] + (int)ImIn[ii + 2] + 2 * (int)ImIn[ii + 3] + 5 * (int)ImIn[ii + 4];
	ii += w;
	val += 4 * (int)ImIn[ii] + (int)ImIn[ii + 1] + (int)ImIn[ii + 3] + 4 * (int)ImIn[ii + 4];
	ii += w;
	val += 5 * (int)ImIn[ii] + 2 * (int)ImIn[ii + 1] + (int)ImIn[ii + 2] + 2 * (int)ImIn[ii + 3] + 5 * (int)ImIn[ii + 4];
	ii += w;
	val += 8 * (int)ImIn[ii] + 5 * (int)ImIn[ii + 1] + 4 * (int)ImIn[ii + 2] + 5 * (int)ImIn[ii + 3] + 8 * (int)ImIn[ii + 4];
	ImOut[i] = (uint16_t)(val / 100);
}

static dim3 Grid(int w, int h) { return dim3((unsigned)max(1, (w - 4 + 31) / 32), (unsigned)max(1, (h - 4 + 7) / 8)); }

cudaError_t Ess(const uint16_t* in, uint16_t* out, int w, int h, cudaStream_t st)
{
	k_ess<<<Grid(w, h), dim3(32, 8), 0, st>>>(in, out, w, h);
	return cudaGetLastError();
}

cudaError_t Ecp(const uint16_t* in, uint16_t* out, int w, int h, cudaStream_t st)
{
	k_ecp<<<Grid(w, h), dim3(32, 8), 0, st>>>(in, out, w, h);
	return cudaGetLastError();
}

}
}
