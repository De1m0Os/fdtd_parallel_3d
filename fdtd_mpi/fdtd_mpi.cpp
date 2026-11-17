#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

inline size_t idx(int i, int j, int k, int N) 
{
	return ((size_t)i * N + j) * N + k;
}

inline float source_pulse(int step, float dt) 
{
	float t = step * dt;
	float t0 = 20.0f * dt;
	float tau = 6.0f * dt;
	float f0 = 0.05f / dt;
	return std::exp(-((t - t0) * (t - t0)) / (2.0f * tau * tau)) * std::sin(2.0f * 3.14159265f * f0 * t);
}

//Последовательный эталон на процессе rank 0
double run_sequential(int N, int steps, std::vector<float>& Ez_out, double& energy_out) 
{
	size_t total = (size_t)N * N * N;
	std::vector<float> Ex(total, 0.0f), Ey(total, 0.0f), Ez(total, 0.0f);
	std::vector<float> Hx(total, 0.0f), Hy(total, 0.0f), Hz(total, 0.0f);
	const float Ch = 0.5f, Ce = 0.5f, dt = 1.0f;
	int ic = N / 2, jc = N / 2, kc = N / 2;

	double t0 = MPI_Wtime();
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
		Ez[idx(ic, jc, kc, N)] += source_pulse(t, dt);
	}
	double t1 = MPI_Wtime();

	double energy = 0.0;
	for (size_t m = 0; m < total; ++m)
		energy += 0.5 * (Ex[m] * Ex[m] + Ey[m] * Ey[m] + Ez[m] * Ez[m] + Hx[m] * Hx[m] + Hy[m] * Hy[m] + Hz[m] * Hz[m]);
	energy_out = energy;
	Ez_out = Ez;
	return t1 - t0;
}

//Вспомогательные функции обновления одного X-слоя
inline void update_H_slice(int i_loc, int i_glob, int N, float Ch, const float* Ex, const float* Ey, const float* Ez, float* Hx, float* Hy, float* Hz) 
{
	if (i_glob < 0 || i_glob >= N - 1) return;
	for (int j = 0; j < N - 1; ++j) 
	{
		for (int k = 0; k < N - 1; ++k) 
		{
			size_t id = idx(i_loc, j, k, N);
			Hx[id] -= Ch * ((Ez[idx(i_loc, j + 1, k, N)] - Ez[id]) - (Ey[idx(i_loc, j, k + 1, N)] - Ey[id]));
			Hy[id] -= Ch * ((Ex[idx(i_loc, j, k + 1, N)] - Ex[id]) - (Ez[idx(i_loc + 1, j, k, N)] - Ez[id]));
			Hz[id] -= Ch * ((Ey[idx(i_loc + 1, j, k, N)] - Ey[id]) - (Ex[idx(i_loc, j + 1, k, N)] - Ex[id]));
		}
	}
}

inline void update_E_slice(int i_loc, int i_glob, int N, float Ce, float* Ex, float* Ey, float* Ez,	const float* Hx, const float* Hy, const float* Hz) 
{
	if (i_glob <= 0 || i_glob >= N - 1) return;
	for (int j = 1; j < N - 1; ++j) 
	{
		for (int k = 1; k < N - 1; ++k) 
		{
			size_t id = idx(i_loc, j, k, N);
			Ex[id] += Ce * ((Hz[id] - Hz[idx(i_loc, j - 1, k, N)]) - (Hy[id] - Hy[idx(i_loc, j, k - 1, N)]));
			Ey[id] += Ce * ((Hx[id] - Hx[idx(i_loc, j, k - 1, N)]) - (Hz[id] - Hz[idx(i_loc - 1, j, k, N)]));
			Ez[id] += Ce * ((Hy[id] - Hy[idx(i_loc - 1, j, k, N)]) - (Hx[id] - Hx[idx(i_loc, j - 1, k, N)]));
		}
	}
}

int main(int argc, char** argv) 
{
	MPI_Init(&argc, &argv);
	int rank = 0, size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int N = (argc > 1) ? std::atoi(argv[1]) : 120;
	int steps = (argc > 2) ? std::atoi(argv[2]) : 100;

	std::vector<float> Ez_ref;
	double energy_ref = 0.0, t_seq = 0.0;
	if (rank == 0) 
	{
		t_seq = run_sequential(N, steps, Ez_ref, energy_ref);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double t_total_start = MPI_Wtime();

	//Декомпозиция области по оси X и рассылка параметров (накладные расходы)
	int params[2] = { N, steps };
	MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD);

	int base_nx = N / size;
	int rem = N % size;
	int local_nx = base_nx + (rank < rem ? 1 : 0);
	int start_x = rank * base_nx + (rank < rem ? rank : rem);

	int plane = N * N;
	//Локальные массивы с 2 теневыми слоями: индекс 0 (левое гало) и local_nx + 1 (правое гало)
	size_t local_total = (size_t)(local_nx + 2) * plane;
	std::vector<float> Ex(local_total, 0.0f), Ey(local_total, 0.0f), Ez(local_total, 0.0f);
	std::vector<float> Hx(local_total, 0.0f), Hy(local_total, 0.0f), Hz(local_total, 0.0f);

	int left_peer = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
	int right_peer = (rank < size - 1) ? (rank + 1) : MPI_PROC_NULL;

	const float Ch = 0.5f, Ce = 0.5f, dt = 1.0f;
	int ic = N / 2, jc = N / 2, kc = N / 2;

	double t_calc_local = 0.0;
	double t_comm_local = 0.0;

	for (int t = 0; t < steps; ++t) 
	{
		double c0 = MPI_Wtime();
		//Обмен гранями Ey, Ez для расчета слоя i_loc = local_nx:
		t_calc_local += (MPI_Wtime() - c0);

		double m0 = MPI_Wtime();
		MPI_Request reqH[4];
		//Отправляем свой левый слой i_loc=1 соседу слева в его правое гало (local_nx+1)
		MPI_Isend(&Ey[1 * plane], plane, MPI_FLOAT, left_peer, 10, MPI_COMM_WORLD, &reqH[0]);
		MPI_Isend(&Ez[1 * plane], plane, MPI_FLOAT, left_peer, 11, MPI_COMM_WORLD, &reqH[1]);
		MPI_Irecv(&Ey[(local_nx + 1) * plane], plane, MPI_FLOAT, right_peer, 10, MPI_COMM_WORLD, &reqH[2]);
		MPI_Irecv(&Ez[(local_nx + 1) * plane], plane, MPI_FLOAT, right_peer, 11, MPI_COMM_WORLD, &reqH[3]);
		t_comm_local += (MPI_Wtime() - m0);

		//Пока идет пересылка по сети, считаем внутренние слои H: i_loc = 1 ... local_nx - 1
		c0 = MPI_Wtime();
		for (int i_loc = 1; i_loc <= local_nx - 1; ++i_loc) 
		{
			update_H_slice(i_loc, start_x + i_loc - 1, N, Ch, Ex.data(), Ey.data(), Ez.data(), Hx.data(), Hy.data(), Hz.data());
		}
		t_calc_local += (MPI_Wtime() - c0);

		//Ожидаем прихода правой грани и досчитываем приграничный слой i_loc = local_nx
		m0 = MPI_Wtime();
		MPI_Waitall(4, reqH, MPI_STATUSES_IGNORE);
		t_comm_local += (MPI_Wtime() - m0);

		c0 = MPI_Wtime();
		update_H_slice(local_nx, start_x + local_nx - 1, N, Ch, Ex.data(), Ey.data(), Ez.data(), Hx.data(), Hy.data(), Hz.data());
		t_calc_local += (MPI_Wtime() - c0);

		m0 = MPI_Wtime();
		MPI_Request reqE[4];
		MPI_Isend(&Hy[local_nx * plane], plane, MPI_FLOAT, right_peer, 20, MPI_COMM_WORLD, &reqE[0]);
		MPI_Isend(&Hz[local_nx * plane], plane, MPI_FLOAT, right_peer, 21, MPI_COMM_WORLD, &reqE[1]);
		MPI_Irecv(&Hy[0 * plane], plane, MPI_FLOAT, left_peer, 20, MPI_COMM_WORLD, &reqE[2]);
		MPI_Irecv(&Hz[0 * plane], plane, MPI_FLOAT, left_peer, 21, MPI_COMM_WORLD, &reqE[3]);
		t_comm_local += (MPI_Wtime() - m0);

		//Пока идет пересылка, считаем внутренние слои E: i_loc = 2 ... local_nx
		c0 = MPI_Wtime();
		for (int i_loc = 2; i_loc <= local_nx; ++i_loc) 
		{
			update_E_slice(i_loc, start_x + i_loc - 1, N, Ce, Ex.data(), Ey.data(), Ez.data(), Hx.data(), Hy.data(), Hz.data());
		}
		t_calc_local += (MPI_Wtime() - c0);

		m0 = MPI_Wtime();
		MPI_Waitall(4, reqE, MPI_STATUSES_IGNORE);
		t_comm_local += (MPI_Wtime() - m0);

		c0 = MPI_Wtime();
		update_E_slice(1, start_x, N, Ce, Ex.data(), Ey.data(), Ez.data(), Hx.data(), Hy.data(), Hz.data());

		//Источник в центре области
		if (ic >= start_x && ic < start_x + local_nx) 
		{
			int ic_loc = ic - start_x + 1;
			Ez[idx(ic_loc, jc, kc, N)] += source_pulse(t, dt);
		}
		t_calc_local += (MPI_Wtime() - c0);
	}

	//Сбор результирующего поля Ez на процессе 0 для верификации точности (входит в накладные расходы)
	std::vector<int> recvcounts(size), displs(size);
	for (int r = 0; r < size; ++r) 
	{
		int r_nx = base_nx + (r < rem ? 1 : 0);
		int r_start = r * base_nx + (r < rem ? r : rem);
		recvcounts[r] = r_nx * plane;
		displs[r] = r_start * plane;
	}

	std::vector<float> Ez_gathered;
	if (rank == 0) Ez_gathered.resize((size_t)N * N * N);

	MPI_Gatherv(&Ez[1 * plane], local_nx * plane, MPI_FLOAT,
		Ez_gathered.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
		0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	double t_total = MPI_Wtime() - t_total_start;

	double t_calc_max = 0.0;
	MPI_Reduce(&t_calc_local, &t_calc_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (rank == 0) 
	{
		double t_ovh = (t_total > t_calc_max) ? (t_total - t_calc_max) : 0.0;
		double ovh_pct = (t_ovh / t_total) * 100.0;

		double max_err = 0.0, l2_num = 0.0, l2_den = 0.0;
		for (size_t m = 0; m < Ez_ref.size(); ++m) 
		{
			double diff = std::fabs(Ez_gathered[m] - Ez_ref[m]);
			if (diff > max_err) max_err = diff;
			l2_num += diff * diff;
			l2_den += (double)Ez_ref[m] * Ez_ref[m];
		}

		double s_calc = t_seq / t_calc_max;
		double e_calc = s_calc / size;
		double s_tot = t_seq / t_total;
		double e_tot = s_tot / size;
		double cells_updated = (double)(N - 2) * (N - 2) * (N - 2) * steps;
		double mcells_s = (cells_updated / t_total) / 1e6;
		double bw_gbs = (cells_updated * 96.0) / (t_total * 1e9);

		std::printf(" 3D FDTD Pure MPI (Non-blocking Overlap) | Grid: %d^3 | Steps: %d | MPI Processes: %d | T_seq = %.4f s\n",	N, steps, size, t_seq);
		std::printf("| Prc | T_calc,s | T_ovh,s  | T_tot,s  | Ovh,%%  | S_calc | E_clc | S_tot  | E_tot | MCells/s | GB/s   | L_inf   |\n");
		std::printf("| %3d | %8.4f | %8.4f | %8.4f | %5.1f%% | %6.2f | %5.3f | %6.2f | %5.3f | %8.2f | %6.2f | %.1e |\n", size, t_calc_max, t_ovh, t_total, ovh_pct, s_calc, e_calc, s_tot, e_tot, mcells_s, bw_gbs, max_err);
	}

	MPI_Finalize();
	return 0;
}