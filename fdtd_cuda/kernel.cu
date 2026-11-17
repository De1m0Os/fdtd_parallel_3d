#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include "cuda_runtime.h"

#define TILE_X 16
#define TILE_Y 16

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

//Ядро обновления магнитного поля H с кэшированием среза в разделяемой памяти (__shared__)
__global__ void kernel_update_H_shared(int N, float Ch, const float* __restrict__ Ex, const float* __restrict__ Ey, const float* __restrict__ Ez, float* __restrict__ Hx, float* __restrict__ Hy, float* __restrict__ Hz) 
{
    __shared__ float s_Ex[TILE_Y + 1][TILE_X + 1];
    __shared__ float s_Ey[TILE_Y + 1][TILE_X + 1];
    __shared__ float s_Ez[TILE_Y + 1][TILE_X + 1];

    int k = blockIdx.x * TILE_X + threadIdx.x;
    int j = blockIdx.y * TILE_Y + threadIdx.y;
    int i = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (i < N && j < N && k < N) 
	{
        size_t id = idx(i, j, k, N);
        s_Ex[ty][tx] = Ex[id];
        s_Ey[ty][tx] = Ey[id];
        s_Ez[ty][tx] = Ez[id];

        if (tx == TILE_X - 1 && k + 1 < N) 
		{
            s_Ex[ty][tx + 1] = Ex[idx(i, j, k + 1, N)];
            s_Ey[ty][tx + 1] = Ey[idx(i, j, k + 1, N)];
        }
        if (ty == TILE_Y - 1 && j + 1 < N)
		{
            s_Ex[ty + 1][tx] = Ex[idx(i, j + 1, k, N)];
            s_Ez[ty + 1][tx] = Ez[idx(i, j + 1, k, N)];
        }
    }
    __syncthreads();

    if (i < N - 1 && j < N - 1 && k < N - 1) 
	{
        size_t id = idx(i, j, k, N);
        float ex_ijk = s_Ex[ty][tx];
        float ey_ijk = s_Ey[ty][tx];
        float ez_ijk = s_Ez[ty][tx];

        Hx[id] -= Ch * ((s_Ez[ty + 1][tx] - ez_ijk) - (s_Ey[ty][tx + 1] - ey_ijk));
        Hy[id] -= Ch * ((s_Ex[ty][tx + 1] - ex_ijk) - (Ez[idx(i + 1, j, k, N)] - ez_ijk));
        Hz[id] -= Ch * ((Ey[idx(i + 1, j, k, N)] - ey_ijk) - (s_Ex[ty + 1][tx] - ex_ijk));
    }
}

//Ядро обновления электрического поля E с разделяемой памятью (__shared__)
__global__ void kernel_update_E_shared(int N, float Ce, float* __restrict__ Ex, float* __restrict__ Ey, float* __restrict__ Ez, const float* __restrict__ Hx, const float* __restrict__ Hy, const float* __restrict__ Hz, int step, int ic, int jc, int kc) 
{
    __shared__ float s_Hx[TILE_Y + 1][TILE_X + 1];
    __shared__ float s_Hy[TILE_Y + 1][TILE_X + 1];
    __shared__ float s_Hz[TILE_Y + 1][TILE_X + 1];

    int k = blockIdx.x * TILE_X + threadIdx.x;
    int j = blockIdx.y * TILE_Y + threadIdx.y;
    int i = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if (i < N && j < N && k < N) 
	{
        size_t id = idx(i, j, k, N);
        s_Hx[ty + 1][tx + 1] = Hx[id];
        s_Hy[ty + 1][tx + 1] = Hy[id];
        s_Hz[ty + 1][tx + 1] = Hz[id];

        if (tx == 0 && k > 0) 
		{
            s_Hx[ty + 1][0] = Hx[idx(i, j, k - 1, N)];
            s_Hy[ty + 1][0] = Hy[idx(i, j, k - 1, N)];
        }
        if (ty == 0 && j > 0) 
		{
            s_Hx[0][tx + 1] = Hx[idx(i, j - 1, k, N)];
            s_Hz[0][tx + 1] = Hz[idx(i, j - 1, k, N)];
        }
    }
    __syncthreads();

    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) 
	{
        size_t id = idx(i, j, k, N);
        float hx_ijk = s_Hx[ty + 1][tx + 1];
        float hy_ijk = s_Hy[ty + 1][tx + 1];
        float hz_ijk = s_Hz[ty + 1][tx + 1];

        Ex[id] += Ce * ((hz_ijk - s_Hz[ty][tx + 1]) - (hy_ijk - s_Hy[ty + 1][tx]));
        Ey[id] += Ce * ((hx_ijk - s_Hx[ty + 1][tx]) - (hz_ijk - Hz[idx(i - 1, j, k, N)]));
        float ez_val = Ez[id] + Ce * ((hy_ijk - Hy[idx(i - 1, j, k, N)]) - (hx_ijk - s_Hx[ty][tx + 1]));
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

    auto t0 = std::chrono::high_resolution_clock::now();
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
    auto t1 = std::chrono::high_resolution_clock::now();
    Ez_out = Ez;
    return std::chrono::duration<double>(t1 - t0).count();
}

void run_cuda_benchmark(int N, int steps) 
{
    std::vector<float> Ez_ref;
    double t_seq = run_sequential(N, steps, Ez_ref);

    size_t total = (size_t)N * N * N;
    size_t bytes = total * sizeof(float);
    std::vector<float> h_Ez(total, 0.0f);

    auto t_tot_start = std::chrono::high_resolution_clock::now();

    //Выделение памяти на GPU и копирование Host -> Device (накладные расходы)
    float *d_Ex, *d_Ey, *d_Ez, *d_Hx, *d_Hy, *d_Hz;
    cudaMalloc(&d_Ex, bytes); cudaMalloc(&d_Ey, bytes); cudaMalloc(&d_Ez, bytes);
    cudaMalloc(&d_Hx, bytes); cudaMalloc(&d_Hy, bytes); cudaMalloc(&d_Hz, bytes);

    cudaMemset(d_Ex, 0, bytes); cudaMemset(d_Ey, 0, bytes); cudaMemset(d_Ez, 0, bytes);
    cudaMemset(d_Hx, 0, bytes); cudaMemset(d_Hy, 0, bytes); cudaMemset(d_Hz, 0, bytes);

    dim3 block(TILE_X, TILE_Y, 1);
    dim3 grid((N + TILE_X - 1) / TILE_X, (N + TILE_Y - 1) / TILE_Y, N);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    int ic = N / 2, jc = N / 2, kc = N / 2;

    //Чистое время выполнения ядер на GPU
    cudaEventRecord(ev_start);
    for (int t = 0; t < steps; ++t)
	{
        kernel_update_H_shared<<<grid, block>>>(N, 0.5f, d_Ex, d_Ey, d_Ez, d_Hx, d_Hy, d_Hz);
        kernel_update_E_shared<<<grid, block>>>(N, 0.5f, d_Ex, d_Ey, d_Ez, d_Hx, d_Hy, d_Hz, t, ic, jc, kc);
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms_calc = 0.0f;
    cudaEventElapsedTime(&ms_calc, ev_start, ev_stop);
    double t_calc = ms_calc * 1e-3;

    //Выгрузка результата Device -> Host (входит в накладные расходы)
    cudaMemcpy(h_Ez.data(), d_Ez, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_Ex); cudaFree(d_Ey); cudaFree(d_Ez);
    cudaFree(d_Hx); cudaFree(d_Hy); cudaFree(d_Hz);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    auto t_tot_end = std::chrono::high_resolution_clock::now();
    double t_total = std::chrono::duration<double>(t_tot_end - t_tot_start).count();
    double t_ovh = (t_total > t_calc) ? (t_total - t_calc) : 0.0;
    double ovh_pct = (t_ovh / t_total) * 100.0;

    double max_err = 0.0;
    for (size_t m = 0; m < total; ++m) 
	{
        double diff = std::fabs(h_Ez[m] - Ez_ref[m]);
        if (diff > max_err) max_err = diff;
    }

    double s_calc = t_seq / t_calc;
    double s_tot  = t_seq / t_total;
    double cells_updated = (double)(N - 2) * (N - 2) * (N - 2) * steps;
    double mcells_calc = (cells_updated / t_calc) / 1e6;
    double mcells_tot  = (cells_updated / t_total) / 1e6;
    double bw_gbs = (cells_updated * 96.0) / (t_calc * 1e9);

    std::printf("| %4d^3 | %8.4f | %8.4f | %8.4f | %8.4f | %5.1f%% | %7.2fx | %7.2fx | %9.1f | %8.1f | %6.1f | %.1e |\n", N, t_seq, t_calc, t_ovh, t_total, ovh_pct, s_calc, s_tot, mcells_calc, mcells_tot, bw_gbs, max_err);
}

int main(int argc, char** argv) 
{
	int N = (argc > 1) ? std::atoi(argv[1]) : 120;
	int steps = (argc > 2) ? std::atoi(argv[2]) : 100;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	std::printf("3D FDTD CUDA Benchmark | Device: %s | Grid: %dx%dx%d | Steps: %d\n", prop.name, N, N, N, steps);
	std::printf("Grid     | T_seq(s) | T_calc(s)| T_ovh(s) | T_tot(s) | Ovh,%%   | S_calc  | S_tot   | MCells/s  | GB/s    | L_inf\n");

	run_cuda_benchmark(N, steps);

	return 0;
}