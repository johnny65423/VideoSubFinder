// Shared pieces of the GPU parity tests.
#pragma once

#include <cstdio>
#include <string>

extern int g_checks, g_failed;

#define CHECK(cond) do { g_checks++; if (!(cond)) { g_failed++; printf("  FAILED %s:%d  %s\n", __FILE__, __LINE__, #cond); } } while (0)

// one line per group of comparisons: how many cases, how many differ
inline void Report(const char* name, long long cases, long long bad)
{
	printf("   %-58s : %s  (%lld cases, %lld differ)\n", name, bad ? "DIFFERENT" : "identical", cases, bad);
	g_checks += (int)cases;
	g_failed += (int)bad;
}

// T1: every kernel against the CPU function it replaces (t1_kernels.cpp). Does nothing without USE_CUDA.
void RunT1();
void RunT4();   // t4_pipeline.cpp: GetTransformedImage with and without the GPU path
void RunT7();   // t4_pipeline.cpp: threads, memory, failing allocation
