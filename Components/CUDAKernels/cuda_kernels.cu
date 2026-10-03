
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

// CUDA 13 removed nppGetStreamContext() and the non-_Ctx NPP functions, so the context is filled manually
static NppStreamContext GetNppStreamContext()
{
	NppStreamContext ctx = {};
	int device = 0;

	checkCuda(cudaGetDevice(&device));
	ctx.hStream = 0;
	ctx.nCudaDeviceId = device;
	checkCuda(cudaDeviceGetAttribute(&ctx.nMultiProcessorCount, cudaDevAttrMultiProcessorCount, device));
	checkCuda(cudaDeviceGetAttribute(&ctx.nMaxThreadsPerMultiProcessor, cudaDevAttrMaxThreadsPerMultiProcessor, device));
	checkCuda(cudaDeviceGetAttribute(&ctx.nMaxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, device));
	int shared_mem_per_block = 0;
	checkCuda(cudaDeviceGetAttribute(&shared_mem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device));
	ctx.nSharedMemPerBlock = (size_t)shared_mem_per_block;
	checkCuda(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device));
	checkCuda(cudaDeviceGetAttribute(&ctx.nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device));
	ctx.nStreamFlags = 0;

	return ctx;
}

int NV12_to_BGR(unsigned char *src_y, unsigned char *src_uv, int src_linesize, unsigned char *dst_data, int w, int h, int W, int H)
{
	NppStatus err;
	NppStreamContext npp_ctx = GetNppStreamContext();
	int nSrcPitchCUDA, res = 0;

	Npp8u* device_nv12[2] = { NULL, NULL };
	Npp8u* device_BGR = NULL;
	Npp8u* device_BGR_resized = NULL;

	checkCuda(cudaMalloc(&device_nv12[0], W * H * sizeof(Npp8u)));
	checkCuda(cudaMalloc(&device_nv12[1], ((W * H) / 2) * sizeof(Npp8u)));

	checkCuda(cudaMalloc(&device_BGR, W * H * 3 * sizeof(Npp8u)));

	if (w != W)
	{
		checkCuda(cudaMalloc(&device_BGR_resized, ((w * h) * 3) * sizeof(Npp8u)));
	}

	checkCuda(cudaMemcpy(device_nv12[0], src_y,
		W * H * sizeof(Npp8u), cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(device_nv12[1], src_uv,
		((W * H) / 2) * sizeof(Npp8u), cudaMemcpyHostToDevice));


	//err = nppiNV12ToBGR_8u_P2C3R(device_nv12, W, device_BGR, (W * 3), NppiSize{ W, H });
	//err = nppiNV12ToBGR_709HDTV_8u_P2C3R(device_nv12, W, device_BGR, (W * 3), NppiSize{ W, H });	
	err = nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(device_nv12, W, device_BGR, (W * 3), NppiSize{ W, H }, npp_ctx);

	if (err == NPP_SUCCESS) {
		if (w != W) {
			err = nppiResize_8u_C3R_Ctx(device_BGR, (W * 3), NppiSize{ W, H }, NppiRect{ 0, 0,  W, H }, device_BGR_resized, (w * 3), NppiSize{ w, h }, NppiRect{ 0, 0,  w, h }, NPPI_INTER_LINEAR, npp_ctx);
		}
	}

	if (err == NPP_SUCCESS){
		if (w != W) {
			checkCuda(cudaMemcpy(dst_data, device_BGR_resized,
				((w * h) * 3) * sizeof(Npp8u), cudaMemcpyDeviceToHost));
		}
		else
		{
			checkCuda(cudaMemcpy(dst_data, device_BGR,
				((w * h) * 3) * sizeof(Npp8u), cudaMemcpyDeviceToHost));
		}

		res = 1;
	}

	checkCuda(cudaFree(device_nv12[0]));
	checkCuda(cudaFree(device_nv12[1]));
	checkCuda(cudaFree(device_BGR));
	if (device_BGR_resized)
	{
		checkCuda(cudaFree(device_BGR_resized));
	}
	
	return res;
}
