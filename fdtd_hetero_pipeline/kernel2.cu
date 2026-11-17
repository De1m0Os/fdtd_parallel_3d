#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <omp.h>
#include "cuda_runtime.h"

#define BLOCK_X 16
#define BLOCK_Y 16

__host__ __device__ inline size_t idx(int i, int j, int k, int N) 
{
	return ((size_t)i * N + j) * N + k;
}

__host__ __device__ inline float source_pulse(int step, float dt) 
{
	float t = step * dt;
	float t0 = 20.0f * dt;
	float tau = 6.0f * dt;
	float f0 = 0.05f / dt;
	return expf(-((t - t0) * (t - t0)) / (2.0f * tau * tau)) * sinf(2.0f * 3.14159265f * f0 * t);
}

//GPU-ядро обновления магнитного поля H в подобласти GPU [0 ... gpu_nx - 1]
__global__ void kernel_hetero_H(int gpu_nx, int N, float Ch, const float* __restrict__ Ex, const float* __restrict__ Ey, const float* __restrict__ Ez, float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz) 
{
	int k = blockIdx.x * BLOCK_X + threadIdx.x;
	int j = blockIdx.y * BLOCK_Y + threadIdx.y;
	int i = blockIdx.z;

	if (i < gpu_nx && i < N - 1 && j < N - 1 && k < N - 1) 
	{
		size_t id = idx(i, j, k, N);
		Hx[id] -= Ch * ((Ez[idx(i, j + 1, k, N)] - Ez[id]) - (Ey[idx(i, j, k + 1, N)] - Ey[id]));
		Hy[id] -= Ch * ((Ex[idx(i, j, k + 1, N)] - Ex[id]) - (Ez[idx(i + 1, j, k, N)] - Ez[id]));
		Hz[id] -= Ch * ((Ey[idx(i + 1, j, k, N)] - Ey[id]) - (Ex[idx(i, j + 1, k, N)] - Ex[id]));
	}
}

//GPU-ядро обновления электрического поля E в подобласти GPU [1 ... gpu_nx - 1]
__global__ void kernel_hetero_E(int gpu_nx, int N, float Ce, float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez, const float* __restrict__ Hx, const float* __restrict__ Hy, const float* __restrict__ Hz, int step, int ic, int jc, int kc) 
{
	int k = blockIdx.x * BLOCK_X + threadIdx.x;
	int j = blockIdx.y * BLOCK_Y + threadIdx.y;
	int i = blockIdx.z + 1;

	if (i < gpu_nx && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) 
	{
		size_t id = idx(i, j, k, N);
		Ex[id] += Ce * ((Hz[id] - Hz[idx(i, j - 1, k, N)]) - (Hy[id] - Hy[idx(i, j, k - 1, N)]));
		Ey[id] += Ce * ((Hx[id] - Hx[idx(i, j, k - 1, N)]) - (Hz[id] - Hz[idx(i - 1, j, k, N)]));
		float ez_val = Ez[id] + Ce * ((Hy[id] - Hy[idx(i - 1, j, k, N)]) - (Hx[id] - Hx[idx(i, j - 1, k, N)]));
		if (i == ic && j == jc && k == kc) 
		{
			ez_val += source_pulse(step, 1.0f);
		}
		Ez[id] = ez_val;
	}
}

//Последовательный эталон на CPU
double run_sequential(int N, int steps, std::vector<float>& Ez_out) 
{
	size_t total = (size_t)N * N * N;
	std::vector<float> Ex(total, 0.0f), Ey(total, 0.0f), Ez(total, 0.0f);
	std::vector<float> Hx(total, 0.0f), Hy(total, 0.0f), Hz(total, 0.0f);
	const float Ch = 0.5f, Ce = 0.5f;
	int ic = N / 2, jc = N / 2, kc = N / 2;

	double t0 = omp_get_wtime();
	for (int t = 0; t < steps; ++t) 
	{
		for (int i = 0; i < N - 1; ++i)
			for (int j = 0; j < N - 1; ++j)
				for (int k = 0; k < N - 1; ++k) 
				{
					size_t id = idx(i, j, k, N);
					Hx[id] -= Ch * ((Ez[idx(i, j + 1, k, N)] - Ez[id]) - (Ey[idx(i, j, k + 1, N)] - Ey[id]));
					Hy[id] -= Ch * ((Ex[idx(i, j, k + 1, N)] - Ex[id]) - (Ez[idx(i + 1, j, k, N)] - Ez[id]));
					Hz[id] -= Ch * ((Ey[idx(i + 1, j, k, N)] - Ey[id]) - (Ex[idx(i, j + 1, k, N)] - Ex[id]));
				}
		for (int i = 1; i < N - 1; ++i)
			for (int j = 1; j < N - 1; ++j)
				for (int k = 1; k < N - 1; ++k) 
				{
					size_t id = idx(i, j, k, N);
					Ex[id] += Ce * ((Hz[id] - Hz[idx(i, j - 1, k, N)]) - (Hy[id] - Hy[idx(i, j, k - 1, N)]));
					Ey[id] += Ce * ((Hx[id] - Hx[idx(i, j, k - 1, N)]) - (Hz[id] - Hz[idx(i - 1, j, k, N)]));
					Ez[id] += Ce * ((Hy[id] - Hy[idx(i - 1, j, k, N)]) - (Hx[id] - Hx[idx(i, j - 1, k, N)]));
				}
		Ez[idx(ic, jc, kc, N)] += source_pulse(t, 1.0f);
	}
	Ez_out = Ez;
	return omp_get_wtime() - t0;
}

//Замер и запуск гетерогенного конвейера CPU (OpenMP) + GPU (CUDA Streams + Pinned Memory)
void run_hetero_pipeline(int N, int steps, float cuRatio_in, double t_seq, const std::vector<float>& Ez_ref) 
{
	size_t total = (size_t)N * N * N;
	size_t plane_elems = (size_t)N * N;
	size_t plane_bytes = plane_elems * sizeof(float);

	double t_total_start = omp_get_wtime();

	//Выделение закрепленной памяти хоста (Pinned Memory) для быстрых асинхронных пересылок DMA
	float* h_Ex, * h_Ey, * h_Ez, * h_Hx, * h_Hy, * h_Hz;
	cudaMallocHost(&h_Ex, total * sizeof(float));
	cudaMallocHost(&h_Ey, total * sizeof(float));
	cudaMallocHost(&h_Ez, total * sizeof(float));
	cudaMallocHost(&h_Hx, total * sizeof(float));
	cudaMallocHost(&h_Hy, total * sizeof(float));
	cudaMallocHost(&h_Hz, total * sizeof(float));

	std::memset(h_Ex, 0, total * sizeof(float));
	std::memset(h_Ey, 0, total * sizeof(float));
	std::memset(h_Ez, 0, total * sizeof(float));
	std::memset(h_Hx, 0, total * sizeof(float));
	std::memset(h_Hy, 0, total * sizeof(float));
	std::memset(h_Hz, 0, total * sizeof(float));

	float* d_Ex, * d_Ey, * d_Ez, * d_Hx, * d_Hy, * d_Hz;
	cudaMalloc(&d_Ex, total * sizeof(float));
	cudaMalloc(&d_Ey, total * sizeof(float));
	cudaMalloc(&d_Ez, total * sizeof(float));
	cudaMalloc(&d_Hx, total * sizeof(float));
	cudaMalloc(&d_Hy, total * sizeof(float));
	cudaMalloc(&d_Hz, total * sizeof(float));

	cudaMemset(d_Ex, 0, total * sizeof(float));
	cudaMemset(d_Ey, 0, total * sizeof(float));
	cudaMemset(d_Ez, 0, total * sizeof(float));
	cudaMemset(d_Hx, 0, total * sizeof(float));
	cudaMemset(d_Hy, 0, total * sizeof(float));
	cudaMemset(d_Hz, 0, total * sizeof(float));

	cudaStream_t stream_gpu;
	cudaStreamCreate(&stream_gpu);

	//Динамическая автокалибровка коэффициента разделения нагрузки cuRatio (если cuRatio_in < 0)
	float cuRatio = cuRatio_in;
	if (cuRatio < 0.0f) 
	{
		int test_slices = N / 2;
		dim3 b_cal(BLOCK_X, BLOCK_Y, 1);
		dim3 g_cal((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y, test_slices);

		cudaDeviceSynchronize();
		double tg0 = omp_get_wtime();
		kernel_hetero_H << <g_cal, b_cal, 0, stream_gpu >> > (test_slices, N, 0.5f, d_Ex, d_Ey, d_Ez, d_Hx, d_Hy, d_Hz);
		cudaStreamSynchronize(stream_gpu);
		double dt_gpu = omp_get_wtime() - tg0;

		double tc0 = omp_get_wtime();
		#pragma omp parallel for schedule(static)
		for (int i = test_slices; i < N - 1; ++i) 
		{
			for (int j = 0; j < N - 1; ++j) 
			{
				for (int k = 0; k < N - 1; ++k) 
				{
					size_t id = idx(i, j, k, N);
					h_Hx[id] -= 0.5f * ((h_Ez[idx(i, j + 1, k, N)] - h_Ez[id]) - (h_Ey[idx(i, j, k + 1, N)] - h_Ey[id]));
				}
			}
		}
		double dt_cpu = omp_get_wtime() - tc0;

		double v_gpu = 1.0 / (dt_gpu + 1e-9);
		double v_cpu = 1.0 / (dt_cpu + 1e-9);
		cuRatio = (float)(v_gpu / (v_gpu + v_cpu));
		if (cuRatio > 0.92f) cuRatio = 0.88f; //Учет времени обмена пограничным слоем
		if (cuRatio < 0.10f) cuRatio = 0.50f;

		cudaMemset(d_Hx, 0, total * sizeof(float));
		std::memset(h_Hx, 0, total * sizeof(float));
	}

	int gpu_nx = (int)std::round(cuRatio * N);
	if (gpu_nx < 2) gpu_nx = 2;
	if (gpu_nx > N - 2) gpu_nx = N - 2;

	dim3 block(BLOCK_X, BLOCK_Y, 1);
	dim3 grid_H((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y, gpu_nx);
	dim3 grid_E((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y, gpu_nx - 1);

	int ic = N / 2, jc = N / 2, kc = N / 2;
	const float Ch = 0.5f, Ce = 0.5f;

	double t_calc_accum = 0.0;

	//Основной цикл гетерогенных вычислений (асинхронная работа GPU и CPU OpenMP)
	for (int t = 0; t < steps; ++t) 
	{
		double c0 = omp_get_wtime();

		kernel_hetero_H << <grid_H, block, 0, stream_gpu >> > (gpu_nx, N, Ch, d_Ex, d_Ey, d_Ez, d_Hx, d_Hy, d_Hz);

		#pragma omp parallel for schedule(static)
		for (int i = gpu_nx; i < N - 1; ++i) 
		{
			for (int j = 0; j < N - 1; ++j) 
			{
				for (int k = 0; k < N - 1; ++k) 
				{
					size_t id = idx(i, j, k, N);
					h_Hx[id] -= Ch * ((h_Ez[idx(i, j + 1, k, N)] - h_Ez[id]) - (h_Ey[idx(i, j, k + 1, N)] - h_Ey[id]));
					h_Hy[id] -= Ch * ((h_Ex[idx(i, j, k + 1, N)] - h_Ex[id]) - (h_Ez[idx(i + 1, j, k, N)] - h_Ez[id]));
					h_Hz[id] -= Ch * ((h_Ey[idx(i + 1, j, k, N)] - h_Ey[id]) - (h_Ex[idx(i, j + 1, k, N)] - h_Ex[id]));
				}
			}
		}
		cudaStreamSynchronize(stream_gpu);
		t_calc_accum += (omp_get_wtime() - c0);

		//Асинхронный обмен пограничным слоем H (слой gpu_nx - 1 с GPU на CPU для расчета E[gpu_nx])
		size_t border_H = (size_t)(gpu_nx - 1) * plane_elems;
		cudaMemcpyAsync(h_Hy + border_H, d_Hy + border_H, plane_bytes, cudaMemcpyDeviceToHost, stream_gpu);
		cudaMemcpyAsync(h_Hz + border_H, d_Hz + border_H, plane_bytes, cudaMemcpyDeviceToHost, stream_gpu);
		cudaStreamSynchronize(stream_gpu);

		c0 = omp_get_wtime();
		kernel_hetero_E << <grid_E, block, 0, stream_gpu >> > (gpu_nx, N, Ce, d_Ex, d_Ey, d_Ez, d_Hx, d_Hy, d_Hz, t, ic, jc, kc);

		#pragma omp parallel for schedule(static)
		for (int i = gpu_nx; i < N - 1; ++i) 
		{
			for (int j = 1; j < N - 1; ++j) 
			{
				for (int k = 1; k < N - 1; ++k) 
				{
					size_t id = idx(i, j, k, N);
					h_Ex[id] += Ce * ((h_Hz[id] - h_Hz[idx(i, j - 1, k, N)]) - (h_Hy[id] - h_Hy[idx(i, j, k - 1, N)]));
					h_Ey[id] += Ce * ((h_Hx[id] - h_Hx[idx(i, j, k - 1, N)]) - (h_Hz[id] - h_Hz[idx(i - 1, j, k, N)]));
					h_Ez[id] += Ce * ((h_Hy[id] - h_Hy[idx(i - 1, j, k, N)]) - (h_Hx[id] - h_Hx[idx(i, j - 1, k, N)]));
				}
			}
		}
		if (ic >= gpu_nx) 
		{
			h_Ez[idx(ic, jc, kc, N)] += source_pulse(t, 1.0f);
		}
		cudaStreamSynchronize(stream_gpu);
		t_calc_accum += (omp_get_wtime() - c0);

		//Асинхронный обмен пограничным слоем E (слой gpu_nx с CPU на GPU для расчета H[gpu_nx-1])
		size_t border_E = (size_t)gpu_nx * plane_elems;
		cudaMemcpyAsync(d_Ey + border_E, h_Ey + border_E, plane_bytes, cudaMemcpyHostToDevice, stream_gpu);
		cudaMemcpyAsync(d_Ez + border_E, h_Ez + border_E, plane_bytes, cudaMemcpyHostToDevice, stream_gpu);
		cudaStreamSynchronize(stream_gpu);
	}

	//Сборка части сетки с GPU в общую память хоста (входит в накладные расходы)
	cudaMemcpyAsync(h_Ez, d_Ez, (size_t)gpu_nx * plane_bytes, cudaMemcpyDeviceToHost, stream_gpu);
	cudaStreamSynchronize(stream_gpu);

	double t_total = omp_get_wtime() - t_total_start;
	double t_calc = t_calc_accum;
	double t_ovh = (t_total > t_calc) ? (t_total - t_calc) : 0.0;
	double ovh_pct = (t_ovh / t_total) * 100.0;

	double max_err = 0.0;
	for (size_t m = 0; m < total; ++m) 
	{
		double diff = std::fabs(h_Ez[m] - Ez_ref[m]);
		if (diff > max_err) max_err = diff;
	}

	double s_calc = t_seq / t_calc;
	double s_tot = t_seq / t_total;
	double cells_updated = (double)(N - 2) * (N - 2) * (N - 2) * steps;
	double mcells_calc = (cells_updated / t_calc) / 1e6;
	double mcells_tot = (cells_updated / t_total) / 1e6;
	double bw_gbs = (cells_updated * 96.0) / (t_calc * 1e9);

	std::printf("| %5.2f (%3d/%3d) | %8.4f | %8.4f | %8.4f | %5.1f%% | %7.2fx | %7.2fx | %9.1f | %8.1f | %6.1f | %.1e |\n", (float)gpu_nx / N, gpu_nx, N - gpu_nx, t_calc, t_ovh, t_total, ovh_pct, s_calc, s_tot, mcells_calc, mcells_tot, bw_gbs, max_err);

	cudaStreamDestroy(stream_gpu);
	cudaFree(d_Ex); cudaFree(d_Ey); cudaFree(d_Ez);
	cudaFree(d_Hx); cudaFree(d_Hy); cudaFree(d_Hz);
	cudaFreeHost(h_Ex); cudaFreeHost(h_Ey); cudaFreeHost(h_Ez);
	cudaFreeHost(h_Hx); cudaFreeHost(h_Hy); cudaFreeHost(h_Hz);
}

int main(int argc, char** argv) 
{
	int N = (argc > 1) ? std::atoi(argv[1]) : 120;
	int steps = (argc > 2) ? std::atoi(argv[2]) : 100;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	std::vector<float> Ez_ref;
	double t_seq = run_sequential(N, steps, Ez_ref);

	std::printf(" 3D FDTD Heterogeneous CPU (%d OMP threads) + GPU (%s) Pipeline | Grid: %d^3 | Steps: %d | T_seq = %.4f s\n", omp_get_max_threads(), prop.name, N, steps, t_seq);
	std::printf("| cuRatio (GPU/CPU)| T_calc,s | T_ovh,s  | T_tot,s  | Ovh,%%  |  S_calc  |  S_tot   | MCells_cl | MCells_t | GB/s   | L_inf   |\n");

	if (argc > 3) 
	{
		float r = (float)std::atof(argv[3]);
		run_hetero_pipeline(N, steps, r, t_seq, Ez_ref);
	}
	else 
	{
		float ratios[] = { -1.0f, 0.50f, 0.70f, 0.80f, 0.85f, 0.90f };
		for (float r : ratios) 
		{
			run_hetero_pipeline(N, steps, r, t_seq, Ez_ref);
		}
	}
	return 0;
}