// BGR -> YUV, bit exact to cv::cvtColor(COLOR_BGR2YUV) for 8 bit images.
// OpenCV (color_yuv.simd.hpp): yuv_shift = 14, R2Y = 4899, G2Y = 9617, B2Y = 1868, R2VI = 14369, B2UI = 8061, delta = 128 << 14, CV_DESCALE(x, n) = (x + (1 << (n - 1))) >> n
#include "kernels.h"

namespace gpu_transform
{
namespace kernels
{

static __global__ void k_bgr2yuv(const uint8_t* __restrict__ bgr, uint8_t* __restrict__ Y, uint8_t* __restrict__ U, uint8_t* __restrict__ V, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	int B = bgr[3 * i], G = bgr[3 * i + 1], R = bgr[3 * i + 2];
	int y = (B * 1868 + G * 9617 + R * 4899 + (1 << 13)) >> 14;
	int cr = ((R - y) * 14369 + (128 << 14) + (1 << 13)) >> 14;   // V
	int cb = ((B - y) * 8061 + (128 << 14) + (1 << 13)) >> 14;    // U
	Y[i] = (uint8_t)min(max(y, 0), 255);
	U[i] = (uint8_t)min(max(cb, 0), 255);
	V[i] = (uint8_t)min(max(cr, 0), 255);
}

cudaError_t BgrToYuv(const uint8_t* bgr, uint8_t* y, uint8_t* u, uint8_t* v, int n, cudaStream_t st)
{
	if (n <= 0) return cudaSuccess;
	k_bgr2yuv<<<(n + 255) / 256, 256, 0, st>>>(bgr, y, u, v, n);
	return cudaGetLastError();
}

}
}
