
#include <nppi.h>
#include "cuda_kernels.h"

int GetCUDADeviceCount()
{
	int count;
	cudaError_t error = cudaGetDeviceCount(&count);

	if (error == cudaSuccess)
	{
		return count;
	}
	else
	{
		return 0;
	}
}

namespace
{
	// NV12_to_BGR() is called for every video frame from many threads at once. Everything that is expensive to create
	// (CUDA stream, NPP context, device buffers) is kept per thread, so a call only copies data and runs the NPP functions.
	// Every thread has its own non-blocking stream, so the threads do not wait for each other on the default stream.
	struct NV12Workspace
	{
		bool ready = false;
		cudaStream_t stream = NULL;
		NppStreamContext npp_ctx = {};

		Npp8u* nv12 = NULL;
		size_t nv12_size = 0;
		Npp8u* bgr = NULL;
		size_t bgr_size = 0;
		Npp8u* bgr_resized = NULL;
		size_t bgr_resized_size = 0;

		~NV12Workspace()
		{
			// The CUDA runtime may be already unloaded when the object of the main thread is destroyed at process exit,
			// so the errors are ignored here.
			if (nv12) cudaFree(nv12);
			if (bgr) cudaFree(bgr);
			if (bgr_resized) cudaFree(bgr_resized);
			if (stream) cudaStreamDestroy(stream);
		}
	};

	thread_local NV12Workspace t_nv12_ws;

	// CUDA 13 removed nppGetStreamContext() and the non-_Ctx NPP functions, so the context is filled manually
	void InitNV12Workspace(NV12Workspace& ws)
	{
		int device = 0;

		checkCuda(cudaGetDevice(&device));
		checkCuda(cudaStreamCreateWithFlags(&ws.stream, cudaStreamNonBlocking));

		ws.npp_ctx.hStream = ws.stream;
		ws.npp_ctx.nCudaDeviceId = device;
		checkCuda(cudaDeviceGetAttribute(&ws.npp_ctx.nMultiProcessorCount, cudaDevAttrMultiProcessorCount, device));
		checkCuda(cudaDeviceGetAttribute(&ws.npp_ctx.nMaxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device));
		checkCuda(cudaDeviceGetAttribute(&ws.npp_ctx.nMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device));
		int shared_mem_per_block = 0;
		checkCuda(cudaDeviceGetAttribute(&shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device));
		ws.npp_ctx.nSharedMemPerBlock = (size_t)shared_mem_per_block;
		checkCuda(cudaDeviceGetAttribute(&ws.npp_ctx.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device));
		checkCuda(cudaDeviceGetAttribute(&ws.npp_ctx.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device));
		ws.npp_ctx.nStreamFlags = cudaStreamNonBlocking;

		ws.ready = true;
	}

	// grows the device buffer if needed, the buffer is reused as long as the frame size is not bigger
	Npp8u* EnsureDeviceBuffer(Npp8u*& buffer, size_t& buffer_size, size_t needed_size)
	{
		if (needed_size > buffer_size)
		{
			if (buffer)
			{
				checkCuda(cudaFree(buffer));
				buffer = NULL;
				buffer_size = 0;
			}

			checkCuda(cudaMalloc(&buffer, needed_size));
			buffer_size = needed_size;
		}

		return buffer;
	}
}

int NV12_to_BGR(unsigned char *src_y, unsigned char *src_uv, int src_linesize, unsigned char *dst_data, int w, int h, int W, int H)
{
	NppStatus err;
	int res = 0;

	NV12Workspace& ws = t_nv12_ws;
	if (!ws.ready)
	{
		InitNV12Workspace(ws);
	}

	const size_t y_size = (size_t)W * H;
	const size_t uv_size = y_size / 2;

	Npp8u* device_nv12[2];
	device_nv12[0] = EnsureDeviceBuffer(ws.nv12, ws.nv12_size, y_size + uv_size);
	device_nv12[1] = device_nv12[0] + y_size;

	Npp8u* device_BGR = EnsureDeviceBuffer(ws.bgr, ws.bgr_size, y_size * 3);
	Npp8u* device_BGR_resized = NULL;

	if (w != W)
	{
		device_BGR_resized = EnsureDeviceBuffer(ws.bgr_resized, ws.bgr_resized_size, ((size_t)w * h) * 3);
	}

	checkCuda(cudaMemcpyAsync(device_nv12[0], src_y, y_size, cudaMemcpyHostToDevice, ws.stream));
	checkCuda(cudaMemcpyAsync(device_nv12[1], src_uv, uv_size, cudaMemcpyHostToDevice, ws.stream));

	//err = nppiNV12ToBGR_8u_P2C3R_Ctx(device_nv12, W, device_BGR, (W * 3), NppiSize{ W, H }, ws.npp_ctx);
	//err = nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(device_nv12, W, device_BGR, (W * 3), NppiSize{ W, H }, ws.npp_ctx);
	err = nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(device_nv12, W, device_BGR, (W * 3), NppiSize{ W, H }, ws.npp_ctx);

	if (err == NPP_SUCCESS) {
		if (w != W) {
			err = nppiResize_8u_C3R_Ctx(device_BGR, (W * 3), NppiSize{ W, H }, NppiRect{ 0, 0,  W, H }, device_BGR_resized, (w * 3), NppiSize{ w, h }, NppiRect{ 0, 0,  w, h }, NPPI_INTER_LINEAR, ws.npp_ctx);
		}
	}

	if (err == NPP_SUCCESS){
		checkCuda(cudaMemcpyAsync(dst_data, (w != W) ? device_BGR_resized : device_BGR,
			((size_t)w * h) * 3, cudaMemcpyDeviceToHost, ws.stream));

		res = 1;
	}

	// wait for the whole chain: after this the device buffers can be reused by the next call of this thread
	checkCuda(cudaStreamSynchronize(ws.stream));

	return res;
}
