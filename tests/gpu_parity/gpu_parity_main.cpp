// Framework of the GPU parity tests.
//
// Every check compares the GPU path with the CPU functions of the project, byte for byte.
// T1 (t1_kernels.cpp) covers every kernel and the pipeline stages; the end-to-end levels T4..T7 of
// claudedocs/gpu_integration_design.md follow with M3. Usage: gpu_parity [--video <path to a test video>]
#include "IPAlgorithms.h"
#include "gpu_transform.h"
#include "parity.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// the report log of the application (DataTypes.h declares it, the main program defines it)
wxString g_ReportFileName = wxT("./gpu_parity_report.log");

int g_checks = 0, g_failed = 0;


static void test_interface_contract()
{
	printf("interface contract\n");
	const int w = 64, h = 48;
	std::vector<uint8_t> bgr(w * h * 3, 0), ff(w * h, 7), ne(w * h, 7), y(w * h, 7);
	int LB[] = { 4 }, LE[] = { 20 };
	gpu_transform::Input in{ bgr.data(), w, h, LB, LE, 1, 0.25, 0.25 };
	gpu_transform::Output out{ ff.data(), ne.data(), y.data() };

	gpu_transform::Status st = gpu_transform::Run(in, out);
	// Output may only be written when the status is Ok; until the kernels are integrated the status is never Ok
	if (st != gpu_transform::Status::Ok)
	{
		CHECK(ff[0] == 7 && ne[0] == 7 && y[0] == 7);
		CHECK(ff.back() == 7 && ne.back() == 7 && y.back() == 7);
	}
	gpu_transform::Stats s = gpu_transform::GetStats();
	CHECK(s.calls >= s.ok + s.fallbacks || s.calls == 0);
	gpu_transform::ReleaseThreadResources();
}

// GetImFFFinalize() is the last step of GetImFF(): ImSF = copy of ImFF, LB / LE aligned to g_segh, tail of ImSF trimmed
static void test_finalize()
{
	printf("GetImFFFinalize\n");
	const int saved_segh = g_segh;
	g_segh = 3;
	const int w = 16, h = 60;

	{   // alignment: LB rounded down, LE rounded up to a multiple of g_segh
		simple_buffer<u8> ff(w * h, (u8)255), sf(w * h, (u8)0);
		simple_buffer<int> LB(2, 0), LE(2, 0);
		LB[0] = 7; LE[0] = 20; LB[1] = 31; LE[1] = 40;
		GetImFFFinalize(ff, sf, LB, LE, 2, w, h);
		CHECK(LB[0] == 6 && LE[0] == 21);
		CHECK(LB[1] == 30 && LE[1] == 42);
		CHECK(memcmp(ff.m_pData, sf.m_pData, w * h) == 0);   // no trimming, copies are equal
	}
	{   // the last band reaches the bottom: LE is moved up to h - g_segh and ImSF below it is cleared
		simple_buffer<u8> ff(w * h, (u8)255), sf(w * h, (u8)0);
		simple_buffer<int> LB(1, 0), LE(1, 0);
		LB[0] = 40; LE[0] = h - 1;
		GetImFFFinalize(ff, sf, LB, LE, 1, w, h);
		CHECK(LB[0] == 39 && LE[0] == h - g_segh);
		bool ok = true;
		for (int i = 0; i < w * h; i++)
		{
			const u8 expected = (i < w * (h - g_segh + 1)) ? 255 : 0;
			if (sf[i] != expected) { ok = false; break; }
		}
		CHECK(ok);
		CHECK(ff[w * h - 1] == 255);   // ImFF itself is not modified
	}
	g_segh = saved_segh;
}

int main(int argc, char** argv)
{
	setvbuf(stdout, NULL, _IONBF, 0);   // keep the output when a test crashes
	std::string video;
	for (int i = 1; i + 1 < argc; i++)
		if (strcmp(argv[i], "--video") == 0) video = argv[i + 1];
	if (!video.empty()) printf("video for the end-to-end tests: %s\n", video.c_str());   // used from M3 on

	printf("CUDA available: %s\n", gpu_transform::IsAvailable() ? "yes" : "no");
	test_interface_contract();
	test_finalize();
	RunT1();

	printf("\n%d checks, %d failed\n", g_checks, g_failed);
	return g_failed ? 1 : 0;
}
