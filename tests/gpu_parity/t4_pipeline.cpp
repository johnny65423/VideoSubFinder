// T4: GetTransformedImage() with the GPU path switched on against the same call with it switched off (all five output images and the return value).
// T7: stress and fallback: many threads at once, no growth of device memory, a failing allocation must still give the CPU result.
// The images are synthetic "subtitle" frames: a quiet background and 0..3 lines of high contrast glyph blocks, so the real ColorFiltration()
// finds a varying number of text line bands.
#include "parity.h"

#ifndef USE_CUDA

void RunT4() { printf("T4 skipped: built without USE_CUDA\n"); }
void RunT7() { printf("T7 skipped: built without USE_CUDA\n"); }

#else

#include "IPAlgorithms.h"
#include "gpu_transform.h"
#include "gpu_transform_test.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>

namespace
{

struct Frame
{
	int w = 0, h = 0, W = 0, H = 0;
	std::vector<u8> bgr;
	simple_buffer<u8> Copy() const { simple_buffer<u8> b((int)bgr.size()); memcpy(b.m_pData, bgr.data(), bgr.size()); return b; }
};

struct Result
{
	int ret = 0;
	simple_buffer<u8> ff, sf, tf, ne, y;
	Result(int n) : ff(n, (u8)7), sf(n, (u8)7), tf(n, (u8)7), ne(n, (u8)7), y(n, (u8)7) {}
};

// lines: number of lines of glyphs (0 gives an image without text line candidates)
void MakeFrame(std::mt19937& rng, int w, int h, int lines, Frame& f)
{
	f.w = w; f.h = h; f.W = w; f.H = h * 3;   // the crop is the lower third of the frame, like the default search area
	f.bgr.assign((size_t)w * h * 3, 0);
	const int base = 20 + (int)(rng() % 60);
	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
			for (int c = 0; c < 3; c++) f.bgr[(y * w + x) * 3 + c] = (u8)(base + (x + y) / 64 % 8 + c);   // smooth: no lines in it

	int row = (int)(rng() % std::max(1, h / 6));
	for (int l = 0; l < lines; l++)
	{
		const int ht = std::max(6, h / 12 + (int)(rng() % std::max(1, h / 12)));
		if (row + ht >= h) break;
		int x = (int)(rng() % std::max(1, w / 8));
		while (x < w - 4)
		{
			const int gw = 3 + (int)(rng() % std::max(1, w / 40));
			const int gap = 2 + (int)(rng() % std::max(1, w / 60));
			for (int yy = row; yy < row + ht; yy++)
				for (int xx = x; xx < std::min(w, x + gw); xx++)
				{
					const u8 v = ((xx + yy) & 1) ? 255 : 0;   // strong horizontal variation inside the glyph
					for (int c = 0; c < 3; c++) f.bgr[(yy * w + xx) * 3 + c] = v;
				}
			x += gw + gap;
		}
		row += ht + 30 + (int)(rng() % std::max(1, h / 8));   // more than twice the margin of ColorFiltration() between two lines
	}
}

void RunOne(const Frame& f, bool gpu, Result& r)
{
	g_use_cuda_gpu_transform = gpu;
	simple_buffer<u8> bgr = f.Copy();   // GetTransformedImage may use the buffer as scratch
	r.ret = GetTransformedImage(bgr, r.ff, r.sf, r.tf, r.ne, r.y, f.w, f.h, f.W, f.H, 0, f.w - 1);
}

bool Same(const Result& a, const Result& b, int n)
{
	return a.ret == b.ret && !memcmp(a.ff.m_pData, b.ff.m_pData, n) && !memcmp(a.sf.m_pData, b.sf.m_pData, n) && !memcmp(a.tf.m_pData, b.tf.m_pData, n) &&
		!memcmp(a.ne.m_pData, b.ne.m_pData, n) && !memcmp(a.y.m_pData, b.y.m_pData, n);
}

struct Settings
{
	bool gpu = g_use_cuda_gpu_transform, cuda = g_use_cuda_gpu, show = g_show_results;
	~Settings() { g_use_cuda_gpu_transform = gpu; g_use_cuda_gpu = cuda; g_show_results = show; }
};

}

void RunT4()
{
	printf("T4 GetTransformedImage, GPU path against CPU path\n");
	Settings keep;
	g_use_cuda_gpu = true;
	g_show_results = false;
	g_color_ranges.clear(); g_outline_color_ranges.clear();

	struct Size { int w, h; };
	const std::vector<Size> sizes = { {1920, 324}, {1280, 216}, {640, 120}, {320, 64} };
	std::mt19937 rng(424242);
	long long cases = 0, bad = 0, with_text = 0, multi_band = 0;
	const gpu_transform::Stats before = gpu_transform::GetStats();

	for (const Size& sz : sizes)
	{
		const int n = sz.w * sz.h;
		for (int rep = 0; rep < 24; rep++)
		{
			Frame f;
			MakeFrame(rng, sz.w, sz.h, rep % 4, f);   // 0, 1, 2, 3 lines

			{   // how many bands does the real ColorFiltration find?
				simple_buffer<int> LB(sz.h, 0), LE(sz.h, 0);
				int N = 0;
				simple_buffer<u8> copy = f.Copy();
				ColorFiltration(copy, LB, LE, N, sz.w, sz.h);
				if (N > 0) with_text++;
				if (N > 1) multi_band++;
			}

			Result cpu(n), gpu(n);
			RunOne(f, false, cpu);
			RunOne(f, true, gpu);
			cases++;
			if (!Same(cpu, gpu, n)) bad++;
		}
	}
	const gpu_transform::Stats after = gpu_transform::GetStats();
	Report("GetTransformedImage: ImFF ImSF ImTF ImNE ImY and the return value", cases, bad);
	printf("      %lld of %lld images have text line bands, %lld of them more than one band\n", with_text, cases, multi_band);

	// the GPU path must really have been used (otherwise the comparison above would compare the CPU with itself)
	const long long gpu_runs = (long long)(after.ok - before.ok);
	printf("      GPU runs: %lld (fallbacks: %lld)\n", gpu_runs, (long long)(after.fallbacks - before.fallbacks));
	CHECK(gpu_runs == with_text);
	CHECK(after.fallbacks == before.fallbacks);

	gpu_transform::ReleaseResources();
}

void RunT7()
{
	printf("T7 stress and fallback\n");
	Settings keep;
	g_use_cuda_gpu = true;
	g_show_results = false;
	g_color_ranges.clear(); g_outline_color_ranges.clear();
	gpu_transform::ReleaseResources();

	// a set of frames with their CPU results
	const int w = 1280, h = 216, n = w * h;
	std::mt19937 rng(777);
	std::vector<Frame> frames(12);
	std::vector<std::unique_ptr<Result>> expected;
	for (size_t i = 0; i < frames.size(); i++)
	{
		MakeFrame(rng, w, h, 1 + (int)(i % 3), frames[i]);
		expected.emplace_back(new Result(n));
		RunOne(frames[i], false, *expected.back());
	}

	{   // many threads at the same time, every result is compared with the CPU
		const int threads = 12, rounds = 12;
		std::atomic<long long> bad(0), done(0);
		const gpu_transform::Stats before = gpu_transform::GetStats();
		auto worker = [&](int id)
		{
			Result r(n);
			for (int k = 0; k < rounds; k++)
			{
				const size_t i = (size_t)(id + k) % frames.size();
				g_use_cuda_gpu_transform = true;
				simple_buffer<u8> bgr = frames[i].Copy();
				r.ret = GetTransformedImage(bgr, r.ff, r.sf, r.tf, r.ne, r.y, w, h, frames[i].W, frames[i].H, 0, w - 1);
				done++;
				if (!Same(r, *expected[i], n)) bad++;
			}
		};
		g_use_cuda_gpu_transform = true;
		std::vector<std::thread> pool;
		for (int t = 0; t < threads; t++) pool.emplace_back(worker, t);
		for (auto& t : pool) t.join();
		const gpu_transform::Stats after = gpu_transform::GetStats();
		Report("12 threads x 12 frames at the same time", done, bad);
		printf("      GPU runs: %lld (fallbacks: %lld), device resource sets: %d (created %llu)\n", (long long)(after.ok - before.ok), (long long)(after.fallbacks - before.fallbacks), after.contexts, (unsigned long long)after.contexts_created);
		CHECK(after.contexts <= threads);   // the pool grows only up to the number of simultaneous calls
		gpu_transform::ReleaseResources();
		CHECK(gpu_transform::GetStats().contexts == 0);
	}

	{   // device memory must not grow while the same thread works again and again
		g_use_cuda_gpu_transform = true;
		Result r(n);
		auto run = [&](size_t i)
		{
			simple_buffer<u8> bgr = frames[i].Copy();
			r.ret = GetTransformedImage(bgr, r.ff, r.sf, r.tf, r.ne, r.y, w, h, frames[i].W, frames[i].H, 0, w - 1);
		};
		for (size_t i = 0; i < frames.size(); i++) run(i);   // warm up: the buffers of this thread exist now
		const size_t free_before = gpu_transform::testing::FreeDeviceBytes();
		for (int k = 0; k < 150; k++) run((size_t)k % frames.size());
		const size_t free_after = gpu_transform::testing::FreeDeviceBytes();
		printf("      free device memory before / after 150 more frames: %zu / %zu bytes\n", free_before, free_after);
		CHECK(free_after + (4u << 20) >= free_before);   // 4 MB of tolerance for other users of the GPU
		gpu_transform::ReleaseResources();
	}

	{   // a failing allocation: the call must give the CPU result and the next call must work again
		g_use_cuda_gpu_transform = true;
		gpu_transform::ReleaseResources();
		gpu_transform::testing::InjectAllocationFailure(true);
		const gpu_transform::Stats before = gpu_transform::GetStats();
		Result r(n);
		simple_buffer<u8> bgr = frames[0].Copy();
		r.ret = GetTransformedImage(bgr, r.ff, r.sf, r.tf, r.ne, r.y, w, h, frames[0].W, frames[0].H, 0, w - 1);
		gpu_transform::testing::InjectAllocationFailure(false);
		const gpu_transform::Stats after = gpu_transform::GetStats();
		CHECK(after.fallbacks == before.fallbacks + 1);
		CHECK(Same(r, *expected[0], n));

		Result r2(n);
		simple_buffer<u8> bgr2 = frames[0].Copy();
		r2.ret = GetTransformedImage(bgr2, r2.ff, r2.sf, r2.tf, r2.ne, r2.y, w, h, frames[0].W, frames[0].H, 0, w - 1);
		CHECK(gpu_transform::GetStats().ok == after.ok + 1);   // recovered
		CHECK(Same(r2, *expected[0], n));
		printf("      failing allocation: CPU result used, next frame back on the GPU\n");
		gpu_transform::ReleaseResources();
	}
}

#endif
