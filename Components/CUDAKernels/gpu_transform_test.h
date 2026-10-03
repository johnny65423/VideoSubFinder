// Entry points for tests/gpu_parity: every stage of the GPU "transform" step with host pointers (upload, run, download, blocking).
// No CUDA types here. Only available when the program is built with USE_CUDA and linked with CUDAKernels.
//
// Every function returns 0 on success and the CUDA error code otherwise.
// Output arrays are copied to the device before the run, so pixels that a kernel does not write (borders) keep the content the caller gave,
// exactly as the CPU functions leave the content of their output buffers.
#pragma once

#include <stdint.h>
#include <stddef.h>

namespace gpu_transform
{
namespace testing
{

// cv::cvtColor(COLOR_BGR2YUV): bgr has w * h * 3 bytes
int BgrToYuv(const uint8_t* bgr, int w, int h, uint8_t* y, uint8_t* u, uint8_t* v);

// kind 'M': ImprovedSobelMEdge, 'N': FastImprovedSobelNEdge, 'H': FastImprovedSobelHEdge
int Sobel(char kind, const uint8_t* in, int w, int h, uint16_t* out);

// FindAndApplyLocalThresholding(im, w, 32, w, h)
int LocalThreshold(uint16_t* im, int w, int h);

// AplyESS(in, ess, w, h) and AplyECP(in, ecp, w, h)
int Convolutions(const uint16_t* in, int w, int h, uint16_t* ess, uint16_t* ecp);

// GetImNE / GetImHE with g_mnthr = mnthr
int GetImNE(const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h, double mnthr, uint8_t* out);
int GetImHE(const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h, double mnthr, uint8_t* out);

// GetImCMOEWithThr1 (variant 0) / GetImCMOEWithThr2 (variant 1) on images of w x h pixels. Band k starts at row row0[k] and has rows[k] rows
// (the CPU functions get the offsets row0[k] * w and the heights rows[k]).
int GetImCMOE(int variant, const uint16_t* moeY, const uint16_t* moeU, const uint16_t* moeV, int w, int h, int N, const int* row0, const int* rows, double mthr, uint16_t* out);

// GetImFF up to its last step: ff is ImFF before GetImFFFinalize() (LB / LE are the values before the alignment). N, LB, LE must be valid bands.
int GetImFF(const uint8_t* y, const uint8_t* u, const uint8_t* v, int w, int h, int N, const int* LB, const int* LE, double mthr, uint8_t* ff);

// Fault injection and resource checks for the stress tests (T7)
void InjectAllocationFailure(bool on);                 // while on, a thread that needs new device buffers gets cudaErrorMemoryAllocation
size_t FreeDeviceBytes();                              // cudaMemGetInfo

// The whole step from the BGR image: ImFF (before GetImFFFinalize), ImNE | ImHE and ImY
int Transform(const uint8_t* bgr, int w, int h, int N, const int* LB, const int* LE, double mthr, double mnthr, uint8_t* ff, uint8_t* ne, uint8_t* y);

}
}
