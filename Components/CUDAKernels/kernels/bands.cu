// Gather / scatter of the rows of the text line bands (the copies GetImFF does with copy_data and with its ImFF[(w * LB[k]) + j] loop).
// Plain device to device copies: the rows of a band are contiguous.
#include "kernels.h"

namespace gpu_transform
{
namespace kernels
{

cudaError_t GatherBands(const uint8_t* full, uint8_t* gathered, int w, int N, const int* LB, const int* LE, cudaStream_t st)
{
	size_t dst = 0;
	for (int k = 0; k < N; k++)
	{
		size_t cnt = (size_t)(LE[k] - LB[k] + 1) * w;
		cudaError_t e = cudaMemcpyAsync(gathered + dst, full + (size_t)LB[k] * w, cnt, cudaMemcpyDeviceToDevice, st);
		if (e != cudaSuccess) return e;
		dst += cnt;
	}
	return cudaSuccess;
}

cudaError_t ScatterBands(const uint8_t* gathered, uint8_t* full, int w, int h, int N, const int* LB, const int* LE, cudaStream_t st)
{
	cudaError_t e = cudaMemsetAsync(full, 0, (size_t)w * h, st);
	if (e != cudaSuccess) return e;
	size_t src = 0;
	for (int k = 0; k < N; k++)
	{
		size_t cnt = (size_t)(LE[k] - LB[k] + 1) * w;
		e = cudaMemcpyAsync(full + (size_t)LB[k] * w, gathered + src, cnt, cudaMemcpyDeviceToDevice, st);
		if (e != cudaSuccess) return e;
		src += cnt;
	}
	return cudaSuccess;
}

}
}
