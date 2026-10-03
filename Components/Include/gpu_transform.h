// Public interface of the GPU implementation of the "transform" step of GetTransformedImage().
//
// Location         : Components/Include/gpu_transform.h
// Used by          : Components/IPAlgorithms/IPAlgorithms.cpp (GetTransformedImage)
// Implementation   : Components/CUDAKernels/gpu_transform.cu (+ kernels/*.cu), built by the CUDAKernels project (USE_CUDA)
//
// Rules of this interface
//   * No CUDA types and no wx types here, the header can be included by every project of the solution.
//   * The call is blocking and thread safe: any thread may call it. The device resources (stream, buffers) come from a pool and are reused by the next call, whatever thread makes it.
//   * The result is bit identical to the CPU code, or the call reports a status other than Ok and the caller runs the CPU code.
//   * Without USE_CUDA the inline stubs below make every call report Status::Unavailable, no #ifdef is needed at the call sites.
#pragma once

#include <cstdint>
#include <string>

namespace gpu_transform
{

// What the CPU code of GetTransformedImage() has when it reaches the heavy part:
//   ImBGR already contains the crop (output of CVideo::ConvertToBGR), ColorFiltration() has found N text line bands.
struct Input
{
	const uint8_t* bgr;      // w * h * 3, BGR, as used by cv::cvtColor(.., COLOR_BGR2YUV)
	int w;                   // crop width
	int h;                   // crop height
	const int* LB;           // N values: first row of every text line band (inclusive), as returned by ColorFiltration()
	const int* LE;           // N values: last row of every text line band (inclusive)
	int N;                   // number of bands, N >= 1 (N == 0 is handled by the caller before)
	double mthr;             // g_mthr   : moderate threshold used by GetImFF
	double mnthr;            // g_mnthr  : moderate threshold used by GetImNE and GetImHE
};

// Buffers of the caller, w * h bytes each. Written only when Run() returns Status::Ok.
struct Output
{
	uint8_t* ImFF;           // result of GetImFF() before its last step (the alignment of LB / LE to g_segh stays on the CPU)
	uint8_t* ImNE;           // ImNE | ImHE  (GetImNE + GetImHE + CombineTwoImages)
	uint8_t* ImY;            // Y plane of cv::cvtColor(BGR2YUV), GetTransformedImage() returns it to its callers
};

enum class Status
{
	Ok = 0,
	Unavailable,             // built without CUDA, no CUDA device, or switched off
	UnsupportedSize,         // the input is something ColorFiltration() never produces (invalid text line bands, no image, absurd size)
	OutOfMemory,             // cudaMalloc failed
	DeviceError              // a CUDA call failed, the device resources of that call were discarded and the error state cleared
};

struct Stats                 // process wide counters, only for the log
{
	uint64_t calls = 0;
	uint64_t ok = 0;
	uint64_t fallbacks = 0;  // calls that returned a status other than Ok
	uint64_t bytes_to_device = 0;
	uint64_t bytes_from_device = 0;
	int contexts = 0;        // device resource sets that exist now (the pool grows up to the number of simultaneous calls)
	uint64_t contexts_created = 0;   // device resource sets created since the start (should stay close to the number of simultaneous calls)
};

#ifdef USE_CUDA

// true if a CUDA device is present (checked once). Does not allocate anything.
bool IsAvailable();

// The step itself. Blocking, thread safe, bit identical to the CPU code when the result is Ok.
Status Run(const Input& in, const Output& out);

// Frees the idle device resources of the pool (e.g. after a search). Optional: the next call allocates again.
void ReleaseResources();

Stats GetStats();

// The text of the first failure since the last call (empty if there was none). The caller writes it to its log; later failures are only counted in Stats.
std::string TakeFirstErrorMessage();

#else

inline bool IsAvailable() { return false; }
inline Status Run(const Input&, const Output&) { return Status::Unavailable; }
inline void ReleaseResources() {}
inline Stats GetStats() { return Stats(); }
inline std::string TakeFirstErrorMessage() { return std::string(); }

#endif

}
