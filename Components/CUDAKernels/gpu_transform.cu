// Skeleton of the GPU implementation of the "transform" step (see claudedocs/gpu_integration_design.md, milestone M1).
// Run() does not process anything yet: it always reports Status::Unavailable, so the callers keep using the CPU code.
// this file is the implementation: it must see the declarations of the USE_CUDA build whatever the project defines
#ifndef USE_CUDA
#define USE_CUDA
#endif
#include "../Include/gpu_transform.h"
#include <cuda_runtime.h>
#include <atomic>

namespace gpu_transform
{

static std::atomic<uint64_t> g_calls(0), g_fallbacks(0);

bool IsAvailable()
{
	static const bool available = []
	{
		int count = 0;
		return (cudaGetDeviceCount(&count) == cudaSuccess) && (count > 0);
	}();
	return available;
}

Status Run(const Input&, const Output&)
{
	g_calls++;
	g_fallbacks++;
	return Status::Unavailable;
}

void ReleaseThreadResources() {}

Stats GetStats()
{
	Stats s;
	s.calls = g_calls;
	s.fallbacks = g_fallbacks;
	return s;
}

}
