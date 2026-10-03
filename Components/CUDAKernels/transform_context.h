// Device resources and the pipeline of one calling thread for the GPU "transform" step (internal, CUDA types).
// The public interface is gpu_transform.h. See claudedocs/gpu_integration_design.md, sections 5 and 7.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

namespace gpu_transform
{

class Context
{
public:
	Context() {}
	~Context();                                  // releases everything, never aborts (the CUDA runtime may be gone already when a thread ends)
	Context(const Context&) = delete;
	Context& operator=(const Context&) = delete;

	// creates the stream. Safe to call again after a failure.
	cudaError_t Init();
	// makes the buffers large enough for w x h images. They only grow: a smaller image reuses the buffers.
	cudaError_t Reserve(int w, int h);

	cudaStream_t Stream() const { return st_; }
	int Width() const { return w_; }
	int Height() const { return h_; }

	// What ColorFiltration() guarantees about the text line bands: 1 <= N <= MAX_BANDS, ascending, not overlapping, inside [0, h - 1].
	// The device buffers rely on it (the rows of all bands together are at most h).
	static bool BandsValid(int h, int N, const int* LB, const int* LE);

	// ---- pipeline stages. All of them enqueue work on the stream of the context and return the first error (they never synchronize).
	// Input and output buffers are the public device pointers below; the caller fills bgr (or y, u, v) and reads ff / ne / he.
	cudaError_t ComputeYuv();                                                                // bgr -> y, u, v
	cudaError_t ComputeFF(int N, const int* LB, const int* LE, double mthr);                 // GetImFF up to its last step (alignment of LB / LE stays on the CPU)
	cudaError_t ComputeNE(double mnthr);                                                     // GetImNE
	cudaError_t ComputeHE(double mnthr);                                                     // GetImHE
	// ComputeYuv, ComputeFF, ComputeNE, ComputeHE and ne |= he (CombineTwoImages)
	cudaError_t Transform(int N, const int* LB, const int* LE, double mthr, double mnthr);

	// FindAndApplyLocalThresholding(im, w, 32, w, h) on a device image of the context
	cudaError_t ComputeLocalThreshold(uint16_t* im, int w, int h);

	// One half of GetImCMOEWithThr1 (variant 0) / Thr2 (variant 1) on an image of eh rows: the moe* buffers are the input, the result is in cm[variant].
	// row0 / rows: first row and height of every band inside that image.
	cudaError_t ComputeCmoe(int variant, int eh, int N, const int* row0, const int* rows, double mthr);

	// ---- device buffers (valid after Reserve), n = w * h
	uint8_t *bgr = nullptr;                      // 3 n
	uint8_t *y = nullptr, *u = nullptr, *v = nullptr;
	uint8_t *gy = nullptr, *gu = nullptr, *gv = nullptr;   // rows of the bands, one band after the other
	uint8_t *orband = nullptr;                   // OR of the two CMOE results on the gathered image
	uint8_t *ff = nullptr, *ne = nullptr, *he = nullptr;
	uint16_t *moeY = nullptr, *moeU = nullptr, *moeV = nullptr;   // ImprovedSobelMEdge
	uint16_t *noeY = nullptr, *noeU = nullptr, *noeV = nullptr;   // FastImprovedSobelNEdge / HEdge
	uint16_t *r1 = nullptr, *r2 = nullptr;       // GetImNE / GetImHE
	uint16_t *cm[2] = { nullptr, nullptr };      // GetImCMOEWithThr1 / Thr2
	uint16_t *ess = nullptr, *ecp = nullptr;

private:
	void Free();

	cudaStream_t st_ = nullptr;
	char* block_ = nullptr;                      // one allocation for all buffers
	size_t cap_pixels_ = 0;
	size_t cap_hist_bytes_ = 0;
	int w_ = 0, h_ = 0;
	unsigned* dmax_ = nullptr;                   // maxima of the bands (ApplyModerateThreshold)
	unsigned* hist_ = nullptr;                   // histograms of FindAndApplyLocalThresholding
	int* thr_ = nullptr;
};

}
