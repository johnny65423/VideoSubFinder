// T1: every kernel of the GPU "transform" step against the CPU function of the project it replaces, byte for byte.
// Data: random and structured planes in many sizes (also tiny and odd ones), random but valid text line bands, the 24 bit colour cube.
#include "parity.h"

#ifndef USE_CUDA

void RunT1() { printf("T1 skipped: built without USE_CUDA\n"); }

#else

#include "IPAlgorithms.h"
#include "gpu_transform_test.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

// functions of IPAlgorithms.lib that its header does not declare
void ImprovedSobelMEdge(simple_buffer<u8>& ImIn, simple_buffer<u16>& ImMOE, int w, int h);
void FastImprovedSobelNEdge(simple_buffer<u8>& ImIn, simple_buffer<u16>& ImNOE, int w, int h);
void FastImprovedSobelHEdge(simple_buffer<u8>& ImIn, simple_buffer<u16>& ImHOE, int w, int h);
void GetImFF(simple_buffer<u8>& ImFF, simple_buffer<u8>& ImSF, simple_buffer<u8>& ImYFull, simple_buffer<u8>& ImUFull, simple_buffer<u8>& ImVFull, simple_buffer<int>& LB, simple_buffer<int>& LE, int N, int w, int h, int W, int H, double mthr);
void GetImNE(simple_buffer<u8>& ImNE, simple_buffer<u8>& ImY, simple_buffer<u8>& ImU, simple_buffer<u8>& ImV, int w, int h);
void GetImHE(simple_buffer<u8>& ImHE, simple_buffer<u8>& ImY, simple_buffer<u8>& ImU, simple_buffer<u8>& ImV, int w, int h);
void GetImCMOEWithThr1(simple_buffer<u16>& ImCMOE, simple_buffer<u16>& ImYMOE, simple_buffer<u16>& ImUMOE, simple_buffer<u16>& ImVMOE, int w, int h, int W, int H, simple_buffer<int>& offsets, simple_buffer<int>& dhs, int N, double mthr);
void GetImCMOEWithThr2(simple_buffer<u16>& ImCMOE, simple_buffer<u16>& ImYMOE, simple_buffer<u16>& ImUMOE, simple_buffer<u16>& ImVMOE, int w, int h, int W, int H, simple_buffer<int>& offsets, simple_buffer<int>& dhs, int N, double mthr);

namespace
{

struct Size { int w, h; };

// the CPU convolutions need at least 4 rows (their row loops run from 2 to h - 2). 1920 x 324 is the crop of a 1920 x 1080 video with the default search area
const std::vector<Size> kSizes = { {1920, 324}, {640, 100}, {97, 45}, {64, 33}, {200, 70}, {128, 96}, {33, 31}, {17, 9}, {5, 5}, {200, 5} };

// kinds: 0 noise, 1 stripes, 2 sparse bright points, 3 flat, 4 black, 5 gradient, 6 text-like blocks, 7 only 0 and 255
void FillPlane(std::mt19937& rng, simple_buffer<u8>& p, int w, int h, int kind)
{
	const int n = w * h;
	for (int i = 0; i < n; i++)
	{
		switch (kind)
		{
		case 0: p[i] = (u8)(rng() & 255); break;
		case 1: p[i] = (u8)(((i / 7) % 3 == 0) ? 255 : 0); break;
		case 2: p[i] = (u8)((rng() % 100 < 3) ? 255 : 20); break;
		case 3: p[i] = 128; break;
		case 4: p[i] = 0; break;
		case 5: p[i] = (u8)((i * 13) & 255); break;
		case 7: p[i] = (u8)((rng() & 1) ? 255 : 0); break;
		default: p[i] = 40; break;
		}
	}
	if (kind == 6)
	{
		const int blocks = 3 + (int)(rng() % 20);
		for (int b = 0; b < blocks; b++)
		{
			const int bw = 1 + (int)(rng() % std::max(1, w / 3)), bh = 1 + (int)(rng() % std::max(1, h / 2));
			const int x0 = (int)(rng() % w), y0 = (int)(rng() % h);
			const u8 val = (u8)(rng() & 255);
			for (int y = y0; y < std::min(h, y0 + bh); y++)
				for (int x = x0; x < std::min(w, x0 + bw); x++) p[y * w + x] = val;
		}
		for (int i = 0; i < n; i += 1 + (int)(rng() % 37)) p[i] = (u8)(p[i] + (rng() % 5));   // a little noise
	}
}

constexpr int kKinds = 8;

// u16 images with a density of non zero values and a maximum
void FillU16(std::mt19937& rng, simple_buffer<u16>& p, int n, double density, int maxv)
{
	std::uniform_real_distribution<double> d(0.0, 1.0);
	for (int i = 0; i < n; i++) p[i] = (d(rng) < density) ? (u16)(1 + rng() % maxv) : (u16)0;
}

// bands the way ColorFiltration() returns them: ascending, not overlapping, inside the image, at least min(8, h) rows high.
// span: first row where bands may start (in gathered coordinates row0 of the CMOE test, in image coordinates for GetImFF), 0 gaps are allowed
int RandomBands(std::mt19937& rng, int h, int max_bands, std::vector<int>& LB, std::vector<int>& LE)
{
	LB.clear(); LE.clear();
	const int minlen = std::min(8, h);
	const int N = 1 + (int)(rng() % max_bands);
	int row = (rng() % 3 == 0) ? 0 : (int)(rng() % std::max(1, h / 4));
	for (int k = 0; k < N; k++)
	{
		const int len = minlen + (int)(rng() % std::max(1, h / 3));
		const int e = std::min(h - 1, row + len - 1);
		if (e - row + 1 < minlen) break;
		LB.push_back(row); LE.push_back(e);
		row = e + 1 + (int)(rng() % 4);
	}
	return (int)LB.size();
}

void T1_BgrToYuv(std::mt19937& rng)
{
	long long c = 0, b = 0;
	auto compare = [&](const cv::Mat& bgr)
	{
		const int w = bgr.cols, h = bgr.rows;
		cv::Mat yuv, planes[3];
		cv::cvtColor(bgr, yuv, cv::COLOR_BGR2YUV);
		cv::split(yuv, planes);
		std::vector<uint8_t> y((size_t)w * h), u((size_t)w * h), v((size_t)w * h);
		const int rc = gpu_transform::testing::BgrToYuv(bgr.data, w, h, y.data(), u.data(), v.data());
		c++;
		if (rc != 0) { b++; return; }
		for (int r = 0; r < h; r++)
		{
			if (memcmp(planes[0].ptr(r), &y[(size_t)r * w], w) || memcmp(planes[1].ptr(r), &u[(size_t)r * w], w) || memcmp(planes[2].ptr(r), &v[(size_t)r * w], w)) { b++; return; }
		}
	};

	{   // every one of the 16,777,216 colours
		cv::Mat cube(4096, 4096, CV_8UC3);
		for (int i = 0; i < 4096 * 4096; i++)
		{
			uint8_t* px = cube.data + (size_t)3 * i;
			px[0] = (uint8_t)(i & 255); px[1] = (uint8_t)((i >> 8) & 255); px[2] = (uint8_t)(i >> 16);
		}
		compare(cube);
	}
	for (const Size& sz : kSizes)
		for (int kind = 0; kind < kKinds; kind++)
		{
			cv::Mat img(sz.h, sz.w, CV_8UC3);
			simple_buffer<u8> p(sz.w * sz.h * 3);
			FillPlane(rng, p, sz.w * 3, sz.h, kind);
			memcpy(img.data, p.m_pData, (size_t)sz.w * sz.h * 3);
			compare(img);
		}
	Report("BGR -> YUV  (all 2^24 colours + images of every size)", c, b);
}

void T1_Sobel(std::mt19937& rng)
{
	long long c = 0, b = 0;
	for (const Size& sz : kSizes)
	{
		const int n = sz.w * sz.h;
		simple_buffer<u8> in(n);
		for (int kind = 0; kind < kKinds; kind++)
		{
			FillPlane(rng, in, sz.w, sz.h, kind);
			for (char op : { 'M', 'N', 'H' })
			{
				simple_buffer<u16> cpu(n, (u16)0xABCD), gpu(n, (u16)0xABCD);   // the border keeps this content on both sides
				if (op == 'M') ImprovedSobelMEdge(in, cpu, sz.w, sz.h);
				else if (op == 'N') FastImprovedSobelNEdge(in, cpu, sz.w, sz.h);
				else FastImprovedSobelHEdge(in, cpu, sz.w, sz.h);
				const int rc = gpu_transform::testing::Sobel(op, in.m_pData, sz.w, sz.h, gpu.m_pData);
				c++;
				if (rc != 0 || memcmp(cpu.m_pData, gpu.m_pData, (size_t)n * 2)) b++;
			}
		}
	}
	Report("ImprovedSobelMEdge / FastImprovedSobelNEdge / HEdge", c, b);
}

void T1_LocalThreshold(std::mt19937& rng)
{
	long long c = 0, b = 0;
	auto compare = [&](const simple_buffer<u16>& src, int w, int h)
	{
		const int n = w * h;
		simple_buffer<u16> cpu(src), gpu(src);
		FindAndApplyLocalThresholding(cpu, w, 32, w, h);
		const int rc = gpu_transform::testing::LocalThreshold(gpu.m_pData, w, h);
		c++;
		if (rc != 0 || memcmp(cpu.m_pData, gpu.m_pData, (size_t)n * 2)) b++;
	};
	for (const Size& sz : kSizes)
	{
		const int n = sz.w * sz.h;
		simple_buffer<u16> im(n);
		// synthetic: density x range
		for (int maxv : { 1, 2, 255, 4080, 12240, 44880 })
			for (double dens : { 0.0, 0.02, 0.3, 1.0 })
				for (int rep = 0; rep < 2; rep++)
				{
					FillU16(rng, im, n, dens, maxv);
					compare(im, sz.w, sz.h);
				}
		// realistic: the weighted sums of the Sobel images of structured planes (the input of the real function)
		simple_buffer<u8> py(n), pu(n), pv(n);
		simple_buffer<u16> my(n, (u16)0), mu(n, (u16)0), mv(n, (u16)0);
		for (int kind = 0; kind < kKinds; kind++)
			for (int rep = 0; rep < 2; rep++)
			{
				FillPlane(rng, py, sz.w, sz.h, kind); FillPlane(rng, pu, sz.w, sz.h, (kind + 6) % kKinds); FillPlane(rng, pv, sz.w, sz.h, (kind + 3) % kKinds);
				my.set_values(0, n); mu.set_values(0, n); mv.set_values(0, n);
				ImprovedSobelMEdge(py, my, sz.w, sz.h); ImprovedSobelMEdge(pu, mu, sz.w, sz.h); ImprovedSobelMEdge(pv, mv, sz.w, sz.h);
				for (int mode = 0; mode < 2; mode++)
				{
					im.set_values(0, n);
					for (int y = 1; y < sz.h - 1; y++)
						for (int x = 1; x < sz.w - 1; x++)
						{
							const int i = y * sz.w + x;
							im[i] = (u16)(mode == 0 ? my[i] + mu[i] + mv[i] : my[i] + (mu[i] + mv[i]) * 5);
						}
					compare(im, sz.w, sz.h);
				}
			}
	}
	Report("FindAndApplyLocalThresholding(Im, w, 32, w, h)", c, b);
}

void T1_Convolutions(std::mt19937& rng)
{
	long long c = 0, b = 0;
	for (const Size& sz : kSizes)
	{
		const int n = sz.w * sz.h;
		simple_buffer<u16> in(n), cpu_ess(n), cpu_ecp(n), gpu_ess(n), gpu_ecp(n);
		for (int maxv : { 1, 255, 4080, 44880 })
			for (double dens : { 0.0, 0.1, 0.7, 1.0 })
				for (int rep = 0; rep < 2; rep++)
				{
					FillU16(rng, in, n, dens, maxv);
					cpu_ess.set_values(0, n); cpu_ecp.set_values(0, n); gpu_ess.set_values(0, n); gpu_ecp.set_values(0, n);
					AplyESS(in, cpu_ess, sz.w, sz.h);
					AplyECP(in, cpu_ecp, sz.w, sz.h);
					const int rc = gpu_transform::testing::Convolutions(in.m_pData, sz.w, sz.h, gpu_ess.m_pData, gpu_ecp.m_pData);
					c++;
					if (rc != 0 || memcmp(cpu_ess.m_pData, gpu_ess.m_pData, (size_t)n * 2) || memcmp(cpu_ecp.m_pData, gpu_ecp.m_pData, (size_t)n * 2)) b++;
				}
	}
	Report("AplyESS + AplyECP (5x5 convolutions)", c, b);
}

void T1_NeHe(std::mt19937& rng)
{
	long long c = 0, b_ne = 0, b_he = 0;
	for (const Size& sz : kSizes)
	{
		const int n = sz.w * sz.h;
		simple_buffer<u8> py(n), pu(n), pv(n), cpu(n), gpu(n);
		for (double mt : { 0.0, 0.1, 0.25, 0.3, 0.5, 0.9, 1.0 })
			for (int ky = 0; ky < kKinds; ky++)
				for (int ku = 0; ku < kKinds; ku += 3)
				{
					FillPlane(rng, py, sz.w, sz.h, ky); FillPlane(rng, pu, sz.w, sz.h, ku); FillPlane(rng, pv, sz.w, sz.h, (ky + ku) % kKinds);
					g_mnthr = mt;
					c++;
					cpu.set_values(7, n); gpu.set_values(7, n);
					GetImNE(cpu, py, pu, pv, sz.w, sz.h);
					if (gpu_transform::testing::GetImNE(py.m_pData, pu.m_pData, pv.m_pData, sz.w, sz.h, mt, gpu.m_pData) != 0 || memcmp(cpu.m_pData, gpu.m_pData, n)) b_ne++;
					cpu.set_values(7, n); gpu.set_values(7, n);
					GetImHE(cpu, py, pu, pv, sz.w, sz.h);
					if (gpu_transform::testing::GetImHE(py.m_pData, pu.m_pData, pv.m_pData, sz.w, sz.h, mt, gpu.m_pData) != 0 || memcmp(cpu.m_pData, gpu.m_pData, n)) b_he++;
				}
	}
	Report("GetImNE", c, b_ne);
	Report("GetImHE", c, b_he);
}

void T1_Cmoe(std::mt19937& rng)
{
	long long c = 0, b1 = 0, b2 = 0;
	for (const Size& sz : kSizes)
	{
		const int w = sz.w, h = sz.h, n = w * h;
		simple_buffer<u8> py(n), pu(n), pv(n);
		simple_buffer<u16> my(n), mu(n), mv(n), cpu(n), gpu(n);
		std::vector<int> LB, LE;
		for (int kind = 0; kind < kKinds; kind++)
			for (double mt : { 0.0, 0.25, 0.4, 1.0 })
				for (int rep = 0; rep < 2; rep++)
				{
					FillPlane(rng, py, w, h, kind); FillPlane(rng, pu, w, h, (kind + 6) % kKinds); FillPlane(rng, pv, w, h, (kind + 3) % kKinds);
					my.set_values(0, n); mu.set_values(0, n); mv.set_values(0, n);
					ImprovedSobelMEdge(py, my, w, h); ImprovedSobelMEdge(pu, mu, w, h); ImprovedSobelMEdge(pv, mv, w, h);
					// bands of the gathered image: the rows [row0, row0 + rows)
					const int N = RandomBands(rng, h, 4, LB, LE);
					if (N == 0) continue;
					simple_buffer<int> offsets(N), dhs(N);
					std::vector<int> row0(N), rows(N);
					for (int k = 0; k < N; k++) { row0[k] = LB[k]; rows[k] = LE[k] - LB[k] + 1; offsets[k] = row0[k] * w; dhs[k] = rows[k]; }
					c++;
					cpu.set_values(0x1234, n); gpu.set_values(0x1234, n);
					GetImCMOEWithThr1(cpu, my, mu, mv, w, h, w, h, offsets, dhs, N, mt);
					if (gpu_transform::testing::GetImCMOE(0, my.m_pData, mu.m_pData, mv.m_pData, w, h, N, row0.data(), rows.data(), mt, gpu.m_pData) != 0 || memcmp(cpu.m_pData, gpu.m_pData, (size_t)n * 2)) b1++;
					cpu.set_values(0x1234, n); gpu.set_values(0x1234, n);
					GetImCMOEWithThr2(cpu, my, mu, mv, w, h, w, h, offsets, dhs, N, mt);
					if (gpu_transform::testing::GetImCMOE(1, my.m_pData, mu.m_pData, mv.m_pData, w, h, N, row0.data(), rows.data(), mt, gpu.m_pData) != 0 || memcmp(cpu.m_pData, gpu.m_pData, (size_t)n * 2)) b2++;
				}
	}
	Report("GetImCMOEWithThr1 (with bands)", c, b1);
	Report("GetImCMOEWithThr2 (with bands)", c, b2);
}

void T1_FF(std::mt19937& rng)
{
	long long c = 0, b = 0, rejected = 0, rejected_ok = 0;
	for (const Size& sz : kSizes)
	{
		const int w = sz.w, h = sz.h, n = w * h;
		simple_buffer<u8> py(n), pu(n), pv(n), cpu(n), sf(n), gpu(n);
		std::vector<int> LB, LE;
		for (int kind = 0; kind < kKinds; kind++)
			for (double mt : { 0.0, 0.25, 0.4, 1.0 })
				for (int rep = 0; rep < 2; rep++)
				{
					FillPlane(rng, py, w, h, kind); FillPlane(rng, pu, w, h, (kind + 6) % kKinds); FillPlane(rng, pv, w, h, (kind + 3) % kKinds);
					const int N = RandomBands(rng, h, 6, LB, LE);
					if (N == 0) continue;
					simple_buffer<int> cLB(N), cLE(N);
					for (int k = 0; k < N; k++) { cLB[k] = LB[k]; cLE[k] = LE[k]; }
					c++;
					GetImFF(cpu, sf, py, pu, pv, cLB, cLE, N, w, h, w, h, mt);   // ImFF is what the GPU delivers, LB / LE and ImSF are changed only afterwards
					gpu.set_values(9, n);
					if (gpu_transform::testing::GetImFF(py.m_pData, pu.m_pData, pv.m_pData, w, h, N, LB.data(), LE.data(), mt, gpu.m_pData) != 0 || memcmp(cpu.m_pData, gpu.m_pData, n)) b++;
				}
		// band sets that ColorFiltration() can never return must be refused
		{
			const int LBs[][2] = { {10, 30}, {50, 10}, {0, 0}, {5, -1}, {-1, 0} };
			const int LEs[][2] = { {40, 60}, {60, 20}, {h, 0}, {4, 3}, {10, 0} };
			const int Ns[] = { 2, 2, 1, 1, 1 };
			for (int i = 0; i < 5; i++)
			{
				gpu.set_values(9, n);
				rejected++;
				if (gpu_transform::testing::GetImFF(py.m_pData, pu.m_pData, pv.m_pData, w, h, Ns[i], LBs[i], LEs[i], 0.25, gpu.m_pData) != 0) rejected_ok++;
			}
		}
	}
	Report("GetImFF (gather, Sobel, CMOE x2, OR, scatter) with random bands", c, b);
	Report("invalid band sets are refused", rejected, rejected - rejected_ok);
}

void T1_Transform(std::mt19937& rng)
{
	long long c = 0, b = 0;
	for (const Size& sz : kSizes)
	{
		const int w = sz.w, h = sz.h, n = w * h;
		simple_buffer<u8> bgr(n * 3), py(n), pu(n), pv(n), ff(n), sf(n), ne(n), he(n), gff(n), gne(n), gy(n);
		std::vector<int> LB, LE;
		for (int kind = 0; kind < kKinds; kind++)
			for (int rep = 0; rep < 3; rep++)
			{
				FillPlane(rng, bgr, w * 3, h, kind);
				const int N = RandomBands(rng, h, 5, LB, LE);
				if (N == 0) continue;
				const double mt = (rep == 0) ? 0.4 : 0.25, mnt = (rep == 2) ? 0.5 : 0.3;
				// CPU: the code of GetTransformedImage between ColorFiltration() and FilterTransformedImage()
				{
					cv::Mat cv_bgr(h, w, CV_8UC3, bgr.m_pData), yuv, planes[3];
					cv::cvtColor(cv_bgr, yuv, cv::COLOR_BGR2YUV);
					cv::split(yuv, planes);
					for (int r = 0; r < h; r++) { memcpy(py.m_pData + (size_t)r * w, planes[0].ptr(r), w); memcpy(pu.m_pData + (size_t)r * w, planes[1].ptr(r), w); memcpy(pv.m_pData + (size_t)r * w, planes[2].ptr(r), w); }
				}
				simple_buffer<int> cLB(N), cLE(N);
				for (int k = 0; k < N; k++) { cLB[k] = LB[k]; cLE[k] = LE[k]; }
				g_mnthr = mnt;
				GetImFF(ff, sf, py, pu, pv, cLB, cLE, N, w, h, w, h, mt);
				GetImNE(ne, py, pu, pv, w, h);
				GetImHE(he, py, pu, pv, w, h);
				CombineTwoImages(ne, he, w, h);
				c++;
				const int rc = gpu_transform::testing::Transform(bgr.m_pData, w, h, N, LB.data(), LE.data(), mt, mnt, gff.m_pData, gne.m_pData, gy.m_pData);
				if (rc != 0 || memcmp(ff.m_pData, gff.m_pData, n) || memcmp(ne.m_pData, gne.m_pData, n) || memcmp(py.m_pData, gy.m_pData, n)) b++;
			}
	}
	Report("whole step: BGR -> ImFF, ImNE | ImHE, ImY", c, b);
}

}

void RunT1()
{
	printf("T1 kernels against the CPU functions\n");
	g_use_cuda_gpu = false;
	const double saved_mthr = g_mthr, saved_mnthr = g_mnthr;
	std::mt19937 rng(20260101);
	T1_BgrToYuv(rng);
	T1_Sobel(rng);
	T1_LocalThreshold(rng);
	T1_Convolutions(rng);
	T1_NeHe(rng);
	T1_Cmoe(rng);
	T1_FF(rng);
	T1_Transform(rng);
	g_mthr = saved_mthr; g_mnthr = saved_mnthr;
}

#endif
