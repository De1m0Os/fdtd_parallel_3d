#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <omp.h>

//Индексация 3D-сетки Йи (линеаризация в 1D-массив для кэш-локальности)
inline size_t idx(int i, int j, int k, int N) 
{
	return ((size_t)i * N + j) * N + k;
}

//Гауссов радиоимпульс точечного источника в центре области
inline float source_pulse(int step, float dt) 
{
	float t = step * dt;
	float t0 = 20.0f * dt;
	float tau = 6.0f * dt;
	float f0 = 0.05f / dt;
	return std::exp(-((t - t0) * (t - t0)) / (2.0f * tau * tau)) * std::sin(2.0f * 3.14159265f * f0 * t);
}

//Эталонный последовательный алгоритм 3D FDTD
double run_sequential(int N, int steps, std::vector<float>& Ez_out, double& energy_out) 
{
	size_t total = (size_t)N * N * N;
	std::vector<float> Ex(total, 0.0f), Ey(total, 0.0f), Ez(total, 0.0f);
	std::vector<float> Hx(total, 0.0f), Hy(total, 0.0f), Hz(total, 0.0f);

	const float Ch = 0.5f; //Коэффициент Куранта для магнитного поля
	const float Ce = 0.5f; //Коэффициент Куранта для электрического поля
	const float dt = 1.0f;
	int ic = N / 2, jc = N / 2, kc = N / 2;

	double t_start = omp_get_wtime();

	for (int t = 0; t < steps; ++t) 
	{
		//Обновление магнитного поля H (n + 1/2)
		for (int i = 0; i < N - 1; ++i) 
		{
			for (int j = 0; j < N - 1; ++j) 
			{
				for (int k = 0; k < N - 1; ++k) 
				{
					size_t id = idx(i, j, k, N);
					Hx[id] -= Ch * ((Ez[idx(i, j + 1, k, N)] - Ez[id]) - (Ey[idx(i, j, k + 1, N)] - Ey[id]));
					Hy[id] -= Ch * ((Ex[idx(i, j, k + 1, N)] - Ex[id]) - (Ez[idx(i + 1, j, k, N)] - Ez[id]));
					Hz[id] -= Ch * ((Ey[idx(i + 1, j, k, N)] - Ey[id]) - (Ex[idx(i, j + 1, k, N)] - Ex[id]));
				}
			}
		}
		//Обновление электрического поля E (n + 1)
		for (int i = 1; i < N - 1; ++i) 
		{
			for (int j = 1; j < N - 1; ++j) 
			{
				for (int k = 1; k < N - 1; ++k) 
				{
					size_t id = idx(i, j, k, N);
					Ex[id] += Ce * ((Hz[id] - Hz[idx(i, j - 1, k, N)]) - (Hy[id] - Hy[idx(i, j, k - 1, N)]));
					Ey[id] += Ce * ((Hx[id] - Hx[idx(i, j, k - 1, N)]) - (Hz[id] - Hz[idx(i - 1, j, k, N)]));
					Ez[id] += Ce * ((Hy[id] - Hy[idx(i - 1, j, k, N)]) - (Hx[id] - Hx[idx(i, j - 1, k, N)]));
				}
			}
		}
		//Инжекция импульса источника
		Ez[idx(ic, jc, kc, N)] += source_pulse(t, dt);
	}

	double t_end = omp_get_wtime();

	double energy = 0.0;
	for (size_t m = 0; m < total; ++m) 
	{
		energy += 0.5 * (Ex[m] * Ex[m] + Ey[m] * Ey[m] + Ez[m] * Ez[m] + Hx[m] * Hx[m] + Hy[m] * Hy[m] + Hz[m] * Hz[m]);
	}
	energy_out = energy;
	Ez_out = Ez;
	return t_end - t_start;
}

//Параллельный алгоритм 3D FDTD (OpenMP: единая параллельная область + NUMA First-Touch)
void run_openmp(int N, int steps, int num_threads, double t_seq, const std::vector<float>& Ez_ref) 
{
	omp_set_num_threads(num_threads);
	size_t total = (size_t)N * N * N;

	double t_total_start = omp_get_wtime();

	//Выделение буферов (накладные расходы инициализации)
	float* Ex = (float*)malloc(total * sizeof(float));
	float* Ey = (float*)malloc(total * sizeof(float));
	float* Ez = (float*)malloc(total * sizeof(float));
	float* Hx = (float*)malloc(total * sizeof(float));
	float* Hy = (float*)malloc(total * sizeof(float));
	float* Hz = (float*)malloc(total * sizeof(float));

	const float Ch = 0.5f;
	const float Ce = 0.5f;
	const float dt = 1.0f;
	int ic = N / 2, jc = N / 2, kc = N / 2;

	double t_calc_accum = 0.0;

	//Единая параллельная область для минимизации накладных расходов Fork-Join
	#pragma omp parallel reduction(max:t_calc_accum)
	{
		//Параллельная инициализация страниц памяти (NUMA First-Touch)
		#pragma omp for schedule(static)
		for (int i = 0; i < N; ++i) 
		{
			size_t offset = (size_t)i * N * N;
			std::memset(Ex + offset, 0, N * N * sizeof(float));
			std::memset(Ey + offset, 0, N * N * sizeof(float));
			std::memset(Ez + offset, 0, N * N * sizeof(float));
			std::memset(Hx + offset, 0, N * N * sizeof(float));
			std::memset(Hy + offset, 0, N * N * sizeof(float));
			std::memset(Hz + offset, 0, N * N * sizeof(float));
		}

		double local_calc = 0.0;

		for (int t = 0; t < steps; ++t) 
		{
			double c0 = omp_get_wtime();

			//Обновление магнитного поля H (без неявного барьера до конца блока)
			#pragma omp for schedule(static) nowait
			for (int i = 0; i < N - 1; ++i) 
			{
				for (int j = 0; j < N - 1; ++j) 
				{
					for (int k = 0; k < N - 1; ++k) 
					{
						size_t id = idx(i, j, k, N);
						Hx[id] -= Ch * ((Ez[idx(i, j + 1, k, N)] - Ez[id]) - (Ey[idx(i, j, k + 1, N)] - Ey[id]));
						Hy[id] -= Ch * ((Ex[idx(i, j, k + 1, N)] - Ex[id]) - (Ez[idx(i + 1, j, k, N)] - Ez[id]));
						Hz[id] -= Ch * ((Ey[idx(i + 1, j, k, N)] - Ey[id]) - (Ex[idx(i, j + 1, k, N)] - Ex[id]));
					}
				}
			}
			local_calc += (omp_get_wtime() - c0);

			//Барьерная синхронизация между полушагами H и E (входит в накладные расходы)
			#pragma omp barrier

			double c1 = omp_get_wtime();
			//Обновление электрического поля E
			#pragma omp for schedule(static) nowait
			for (int i = 1; i < N - 1; ++i) 
			{
				for (int j = 1; j < N - 1; ++j) 
				{
					for (int k = 1; k < N - 1; ++k) 
					{
						size_t id = idx(i, j, k, N);
						Ex[id] += Ce * ((Hz[id] - Hz[idx(i, j - 1, k, N)]) - (Hy[id] - Hy[idx(i, j, k - 1, N)]));
						Ey[id] += Ce * ((Hx[id] - Hx[idx(i, j, k - 1, N)]) - (Hz[id] - Hz[idx(i - 1, j, k, N)]));
						Ez[id] += Ce * ((Hy[id] - Hy[idx(i - 1, j, k, N)]) - (Hx[id] - Hx[idx(i, j - 1, k, N)]));
					}
				}
			}
			local_calc += (omp_get_wtime() - c1);

			#pragma omp barrier
			#pragma omp single
			{
				Ez[idx(ic, jc, kc, N)] += source_pulse(t, dt);
			}
		}
		t_calc_accum = local_calc;
	}

	double t_total = omp_get_wtime() - t_total_start;
	double t_calc = t_calc_accum;
	double t_ovh = (t_total > t_calc) ? (t_total - t_calc) : 0.0;
	double ovh_pct = (t_ovh / t_total) * 100.0;

	//Верификация численной точности относительно эталона SEQ
	double max_err = 0.0, l2_num = 0.0, l2_den = 0.0;
	for (size_t m = 0; m < total; ++m) 
	{
		double diff = std::fabs(Ez[m] - Ez_ref[m]);
		if (diff > max_err) max_err = diff;
		l2_num += diff * diff;
		l2_den += (double)Ez_ref[m] * Ez_ref[m];
	}
	double l2_err = (l2_den > 0.0) ? std::sqrt(l2_num / l2_den) : 0.0;

	//Расчет метрик ускорения, эффективности и пропускной способности
	double s_calc = t_seq / t_calc;
	double e_calc = s_calc / num_threads;
	double s_tot = t_seq / t_total;
	double e_tot = s_tot / num_threads;
	double cells_updated = (double)(N - 2) * (N - 2) * (N - 2) * steps;
	double mcells_s = (cells_updated / t_total) / 1e6;
	//За 1 шаг ячейки Йи читается/пишется 24 значения float (96 байт)
	double bw_gbs = (cells_updated * 96.0) / (t_total * 1e9);

	std::printf("| %3d | %8.4f | %8.4f | %8.4f | %5.1f%% | %6.2f | %5.3f | %6.2f | %5.3f | %8.2f | %6.2f | %.1e |\n",
		num_threads, t_calc, t_ovh, t_total, ovh_pct,
		s_calc, e_calc, s_tot, e_tot, mcells_s, bw_gbs, max_err);

	free(Ex); free(Ey); free(Ez);
	free(Hx); free(Hy); free(Hz);
}

int main(int argc, char** argv) 
{
	int N = (argc > 1) ? std::atoi(argv[1]) : 120;
	int steps = (argc > 2) ? std::atoi(argv[2]) : 100;
	int single_threads = (argc > 3) ? std::atoi(argv[3]) : 0;

	std::printf(" 3D FDTD OpenMP Benchmark | Grid: %d x %d x %d | Steps: %d | Max Hardware Threads: %d\n", N, N, N, steps, omp_get_max_threads());

	std::vector<float> Ez_ref;
	double energy_ref = 0.0;
	double t_seq = run_sequential(N, steps, Ez_ref, energy_ref);
	double seq_mcells = ((double)(N - 2) * (N - 2) * (N - 2) * steps / t_seq) / 1e6;

	std::printf(" [SEQ Reference] T_seq = %.4f s | Throughput = %.2f MCells/s | Field Energy W = %.6e\n", t_seq, seq_mcells, energy_ref);
	std::printf("| Thr | T_calc,s | T_ovh,s  | T_tot,s  | Ovh,%%  | S_calc | E_clc | S_tot  | E_tot | MCells/s | GB/s   | L_inf   |\n");

	if (single_threads > 0) 
	{
		run_openmp(N, steps, single_threads, t_seq, Ez_ref);
	}
	else 
	{
		int thread_list[] = { 2, 4, 8};
		int max_t = omp_get_max_threads();
		for (int p : thread_list) 
		{
			if (p <= max_t) 
			{
				run_openmp(N, steps, p, t_seq, Ez_ref);
			}
		}
	}
	return 0;
}