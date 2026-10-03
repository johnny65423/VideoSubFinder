// GPU implementation of the "transform" step of GetTransformedImage() (see claudedocs/gpu_integration_design.md).
//
// Run() takes a Context (stream + device buffers) from a pool, so any thread can call it, also the short-lived worker threads of the application,
// and the device buffers are reused by the next call. Run() is blocking: upload, kernels, download, synchronize.
// Anything unexpected (unsupported input, no memory, a CUDA error) is reported through the Status and the caller runs the CPU code; the output is only written when the status is Ok.
#include "transform_context.h"

// this file is the implementation: it must see the declarations of the USE_CUDA build whatever the project defines
#ifndef USE_CUDA
#define USE_CUDA
#endif
#include "../Include/gpu_transform.h"

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gpu_transform
{

namespace
{

// A Context holds ~55 pixel-sized buffers: a 1920 x 324 crop needs ~34 MB, a full 1920 x 1080 frame ~115 MB.
// This is also the largest number of calls that run on the device at the same time; further calls wait for a free Context.
constexpr int kMaxContexts = 32;
// after this many failures in a row the GPU path stops trying (the caller keeps using the CPU)
constexpr int kMaxConsecutiveFailures = 16;

std::atomic<uint64_t> g_calls(0), g_ok(0), g_fallbacks(0), g_to_device(0), g_from_device(0), g_contexts_created(0);
std::atomic<int> g_failures_in_row(0);
std::mutex g_message_mutex;
std::string g_first_message;

void RememberMessage(const std::string& text)
{
	std::lock_guard<std::mutex> lock(g_message_mutex);
	if (g_first_message.empty()) g_first_message = text;
}

// The pool is never destroyed: at the end of the process the CUDA runtime may be gone before the static destructors run.
struct Pool
{
	std::mutex mutex;
	std::condition_variable available;
	std::vector<Context*> idle;
	int total = 0;   // created and not discarded
};

Pool& GetPool()
{
	static Pool* pool = new Pool();
	return *pool;
}

// Returns a Context with a stream, or nullptr (the error is in e).
Context* Acquire(cudaError_t& e)
{
	Pool& pool = GetPool();
	{
		std::unique_lock<std::mutex> lock(pool.mutex);
		pool.available.wait(lock, [&] { return !pool.idle.empty() || pool.total < kMaxContexts; });
		if (!pool.idle.empty())
		{
			Context* c = pool.idle.back();
			pool.idle.pop_back();
			return c;
		}
		pool.total++;
	}
	std::unique_ptr<Context> c(new Context());
	e = c->Init();
	if (e != cudaSuccess)
	{
		std::lock_guard<std::mutex> lock(pool.mutex);
		pool.total--;
		pool.available.notify_one();
		return nullptr;
	}
	g_contexts_created++;
	return c.release();
}

// broken: the Context saw an error, its buffers are freed instead of reused
void Release(Context* c, bool broken)
{
	Pool& pool = GetPool();
	if (broken) delete c;
	std::lock_guard<std::mutex> lock(pool.mutex);
	if (broken) pool.total--;
	else pool.idle.push_back(c);
	pool.available.notify_one();
}

Status Fail(Status st, const std::string& what, cudaError_t e)
{
	g_fallbacks++;
	g_failures_in_row++;
	if (e != cudaSuccess)
	{
		cudaGetLastError();   // clear the sticky part of the error state of this thread
		RememberMessage("gpu_transform: " + what + ": " + cudaGetErrorString(e) + " (the CPU code is used for this image)");
	}
	else
	{
		RememberMessage("gpu_transform: " + what + " (the CPU code is used for this image)");
	}
	return st;
}

Status StatusOf(cudaError_t e) { return (e == cudaErrorMemoryAllocation) ? Status::OutOfMemory : Status::DeviceError; }

}

bool IsAvailable()
{
	static const bool available = []
	{
		int count = 0;
		return (cudaGetDeviceCount(&count) == cudaSuccess) && (count > 0);
	}();
	return available && g_failures_in_row < kMaxConsecutiveFailures;
}

Status Run(const Input& in, const Output& out)
{
	g_calls++;

	if (!IsAvailable())
	{
		g_fallbacks++;
		return Status::Unavailable;
	}

	if (!in.bgr || !out.ImFF || !out.ImNE || !out.ImY || in.w <= 0 || in.h <= 0 || (long long)in.w * in.h * 3 > 0x7fffffffLL ||
		!Context::BandsValid(in.h, in.N, in.LB, in.LE))
	{
		// not an error of the device: the input is something ColorFiltration() never produces
		g_fallbacks++;
		return Status::UnsupportedSize;
	}

	cudaError_t e = cudaSuccess;
	Context* ctx = Acquire(e);
	if (!ctx) return Fail(StatusOf(e), "cannot create the CUDA stream", e);
	Context& c = *ctx;

	e = c.Reserve(in.w, in.h);
	if (e != cudaSuccess)
	{
		Release(ctx, true);
		return Fail(StatusOf(e), "cannot allocate the device buffers", e);
	}

	const size_t n = (size_t)in.w * in.h;
	cudaStream_t st = c.Stream();

	e = cudaMemcpyAsync(c.bgr, in.bgr, n * 3, cudaMemcpyHostToDevice, st);
	if (e == cudaSuccess) e = c.Transform(in.N, in.LB, in.LE, in.mthr, in.mnthr);
	if (e == cudaSuccess) e = cudaMemcpyAsync(out.ImFF, c.ff, n, cudaMemcpyDeviceToHost, st);
	if (e == cudaSuccess) e = cudaMemcpyAsync(out.ImNE, c.ne, n, cudaMemcpyDeviceToHost, st);
	if (e == cudaSuccess) e = cudaMemcpyAsync(out.ImY, c.y, n, cudaMemcpyDeviceToHost, st);
	cudaError_t s = cudaStreamSynchronize(st);
	if (e == cudaSuccess) e = s;
	if (e != cudaSuccess)
	{
		Release(ctx, true);   // the next call starts with fresh resources
		return Fail(StatusOf(e), "device error", e);
	}
	Release(ctx, false);

	g_ok++;
	g_failures_in_row = 0;
	g_to_device += n * 3;
	g_from_device += n * 3;
	return Status::Ok;
}

void ReleaseResources()
{
	Pool& pool = GetPool();
	std::vector<Context*> idle;
	{
		std::lock_guard<std::mutex> lock(pool.mutex);
		idle.swap(pool.idle);
		pool.total -= (int)idle.size();
	}
	for (Context* c : idle) delete c;
	pool.available.notify_all();
}

Stats GetStats()
{
	Pool& pool = GetPool();
	Stats s;
	s.calls = g_calls;
	s.ok = g_ok;
	s.fallbacks = g_fallbacks;
	s.bytes_to_device = g_to_device;
	s.bytes_from_device = g_from_device;
	{
		std::lock_guard<std::mutex> lock(pool.mutex);
		s.contexts = pool.total;
	}
	s.contexts_created = g_contexts_created;
	return s;
}

std::string TakeFirstErrorMessage()
{
	std::lock_guard<std::mutex> lock(g_message_mutex);
	std::string text;
	text.swap(g_first_message);
	return text;
}

}
