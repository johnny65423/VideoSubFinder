// Internal interface of the kernels of the GPU "transform" step (see claudedocs/gpu_integration_design.md).
//
// Every function is a host launcher: it enqueues work on the given stream and returns the first CUDA error it saw (cudaSuccess otherwise).
// Nothing here synchronizes. All arithmetic is copied from Components/IPAlgorithms/IPAlgorithms.cpp and must stay bit identical to it
// (int16 / int32 intermediate values, the exact truncation and tie breaking rules): tests/gpu_parity compares every function with its CPU original.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

namespace gpu_transform
{
namespace kernels
{

// Number of histogram bins of FindAndApplyLocalThresholding: the same constant as MAX_EDGE_STR in IPAlgorithms.cpp.
// Every value that reaches the thresholding is smaller (see the comments about max_ImCMOE in GetImCMOEWithThr1).
constexpr int MAX_EDGE_STR = 11 * 16 * 256;

// Largest number of text line bands that a launcher accepts (size of the device array with the maxima of the bands)
constexpr int MAX_BANDS = 4096;

inline size_t LocalThresholdHistBytes(int h) { return (size_t)((h + 31) / 32) * MAX_EDGE_STR * sizeof(unsigned); }
inline size_t LocalThresholdThrBytes(int h) { return (size_t)((h + 31) / 32) * sizeof(int); }

// bgr2yuv.cu ---------------------------------------------------------------------------------------------
// cv::cvtColor(COLOR_BGR2YUV) for 8 bit images: n pixels, interleaved BGR in, three planes out
cudaError_t BgrToYuv(const uint8_t* bgr, uint8_t* y, uint8_t* u, uint8_t* v, int n, cudaStream_t st);

// bands.cu -----------------------------------------------------------------------------------------------
// Rows [LB[k], LE[k]] of every band, one band after the other (the gathered image has sum(LE - LB + 1) rows)
cudaError_t GatherBands(const uint8_t* full, uint8_t* gathered, int w, int N, const int* LB, const int* LE, cudaStream_t st);
// Inverse: the full image is cleared first, then every band goes back to its rows
cudaError_t ScatterBands(const uint8_t* gathered, uint8_t* full, int w, int h, int N, const int* LB, const int* LE, cudaStream_t st);

// sobel.cu -----------------------------------------------------------------------------------------------
// ImprovedSobelMEdge, FastImprovedSobelNEdge, FastImprovedSobelHEdge. The 1 pixel border of out is not written.
cudaError_t SobelM(const uint8_t* in, uint16_t* out, int w, int h, cudaStream_t st);
cudaError_t SobelN(const uint8_t* in, uint16_t* out, int w, int h, cudaStream_t st);
cudaError_t SobelH(const uint8_t* in, uint16_t* out, int w, int h, cudaStream_t st);

// local_thr.cu -------------------------------------------------------------------------------------------
// FindAndApplyLocalThresholding(im, w, 32, w, h). hist: LocalThresholdHistBytes(h) bytes, thr: LocalThresholdThrBytes(h) bytes (device scratch)
cudaError_t LocalThreshold(uint16_t* im, int w, int h, unsigned* hist, int* thr, cudaStream_t st);

// conv5.cu -----------------------------------------------------------------------------------------------
// AplyESS / AplyECP: 5x5 convolutions of the interior [2, w - 2) x [2, h - 2); the rest of out is not written
cudaError_t Ess(const uint16_t* in, uint16_t* out, int w, int h, cudaStream_t st);
cudaError_t Ecp(const uint16_t* in, uint16_t* out, int w, int h, cudaStream_t st);

// combine.cu ---------------------------------------------------------------------------------------------
// out = a + b + c (mode 0) or a + (b + c) * 5 (mode 1) for the interior [1, w - 1) x [1, h - 1)
cudaError_t WeightedSum(const uint16_t* a, const uint16_t* b, const uint16_t* c, uint16_t* out, int w, int h, int mode, cudaStream_t st);
// ApplyModerateThreshold(im + row0 * w, mthr, w, rows). dmax: one device unsigned (scratch)
cudaError_t ModerateThreshold(uint16_t* im, int w, int row0, int rows, double mthr, unsigned* dmax, cudaStream_t st);
// BorderClear(cm, 2); cm = (r2 + r3) / 2 for the interior [2, w - 2) x [2, h - 2)
cudaError_t CmoeCombine(uint16_t* cm, const uint16_t* r2, const uint16_t* r3, int w, int h, cudaStream_t st);
// out = (a != 0 || b != 0) ? 255 : 0 for the interior [1, w - 1) x [1, h - 1)   (end of GetImNE / GetImHE)
cudaError_t OrInterior(const uint16_t* a, const uint16_t* b, uint8_t* out, int w, int h, cudaStream_t st);
// out = (a != 0 || b != 0) ? 255 : 0 for n pixels   (end of GetImFF)
cudaError_t OrAll(const uint16_t* a, const uint16_t* b, uint8_t* out, int n, cudaStream_t st);
// CombineTwoImages(ne, he): if (ne == 0 && he != 0) ne = 255
cudaError_t CombineImages(uint8_t* ne, const uint8_t* he, int n, cudaStream_t st);

}
}
