#include <mpi.h>
#include <omp.h>
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

double run_sequential(int N, int steps, std::vector<float>& Ez_out) 
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
	Ez_out = Ez;
	return MPI_Wtime() - t0;
}

void update_H_range_omp(int i_start, int i_end, int start_x, int N, float Ch, const float* Ex, const float* Ey, const float* Ez, float* Hx, float* Hy, float* Hz) 
{
	#pragma omp parallel for schedule(static)
	for (int i_loc = i_start; i_loc <= i_end; ++i_loc) 
	{
		int i_glob = start_x + i_loc - 1;
		if (i_glob < 0 || i_glob >= N - 1) continue;
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
}

void update_E_range_omp(int i_start, int i_end, int start_x, int N, float Ce, float* Ex, float* Ey, float* Ez, const float* Hx, const float* Hy, const float* Hz) 
{
	#pragma omp parallel for schedule(static)
	for (int i_loc = i_start; i_loc <= i_end; ++i_loc) 
	{
		int i_glob = start_x + i_loc - 1;
		if (i_glob <= 0 || i_glob >= N - 1) continue;
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
}

int main(int argc, char** argv) 
{
	int provided = 0;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
	int rank = 0, size = 1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int N = (argc > 1) ? std::atoi(argv[1]) : 120;
	int steps = (argc > 2) ? std::atoi(argv[2]) : 100;
	int threads_per_proc = (argc > 3) ? std::atoi(argv[3]) : omp_get_max_threads();
	omp_set_num_threads(threads_per_proc);

	std::vector<float> Ez_ref;
	double t_seq = 0.0;
	if (rank == 0) 
	{
		t_seq = run_sequential(N, steps, Ez_ref);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	double t_total_start = MPI_Wtime();

	int base_nx = N / size;
	int rem = N % size;
	int local_nx = base_nx + (rank < rem ? 1 : 0);
	int start_x = rank * base_nx + (rank < rem ? rank : rem);
	int plane = N * N;

	size_t local_total = (size_t)(local_nx + 2) * plane;
	std::vector<float> Ex(local_total, 0.0f), Ey(local_total, 0.0f), Ez(local_total, 0.0f);
	std::vector<float> Hx(local_total, 0.0f), Hy(local_total, 0.0f), Hz(local_total, 0.0f);

	int left_peer = (rank > 0) ? (rank - 1) : MPI_PROC_NULL;
	int right_peer = (rank < size - 1) ? (rank + 1) : MPI_PROC_NULL;

	const float Ch = 0.5f, Ce = 0.5f, dt = 1.0f;
	int ic = N / 2, jc = N / 2, kc = N / 2;

	double t_calc_local = 0.0;

	for (int t = 0; t < steps; ++t) {
		//Неблокирующий обмен правым гало для полушага H
		MPI_Request reqH[4];
		MPI_Isend(&Ey[1 * plane], plane, MPI_FLOAT, left_peer, 10, MPI_COMM_WORLD, &reqH[0]);
		MPI_Isend(&Ez[1 * plane], plane, MPI_FLOAT, left_peer, 11, MPI_COMM_WORLD, &reqH[1]);
		MPI_Irecv(&Ey[(local_nx + 1) * plane], plane, MPI_FLOAT, right_peer, 10, MPI_COMM_WORLD, &reqH[2]);
		MPI_Irecv(&Ez[(local_nx + 1) * plane], plane, MPI_FLOAT, right_peer, 11, MPI_COMM_WORLD, &reqH[3]);

		//Внутренняя область H считается всеми потоками OpenMP параллельно с пересылкой
		double c0 = MPI_Wtime();
		if (local_nx > 1) 
		{
			update_H_range_omp(1, local_nx - 1, start_x, N, Ch, Ex.data(), Ey.data(), Ez.data(), Hx.data(), Hy.data(), Hz.data());
		}
		t_calc_local += (MPI_Wtime() - c0);

		MPI_Waitall(4, reqH, MPI_STATUSES_IGNORE);

		c0 = MPI_Wtime();
		update_H_range_omp(local_nx, local_nx, start_x, N, Ch, Ex.data(), Ey.data(), Ez.data(), Hx.data(), Hy.data(), Hz.data());
		t_calc_local += (MPI_Wtime() - c0);

		//Неблокирующий обмен левым гало для полушага E
		MPI_Request reqE[4];
		MPI_Isend(&Hy[local_nx * plane], plane, MPI_FLOAT, right_peer, 20, MPI_COMM_WORLD, &reqE[0]);
		MPI_Isend(&Hz[local_nx * plane], plane, MPI_FLOAT, right_peer, 21, MPI_COMM_WORLD, &reqE[1]);
		MPI_Irecv(&Hy[0 * plane], plane, MPI_FLOAT, left_peer, 20, MPI_COMM_WORLD, &reqE[2]);
		MPI_Irecv(&Hz[0 * plane], plane, MPI_FLOAT, left_peer, 21, MPI_COMM_WORLD, &reqE[3]);

		c0 = MPI_Wtime();
		if (local_nx > 1) 
		{
			update_E_range_omp(2, local_nx, start_x, N, Ce, Ex.data(), Ey.data(), Ez.data(), Hx.data(), Hy.data(), Hz.data());
		}
		t_calc_local += (MPI_Wtime() - c0);

		MPI_Waitall(4, reqE, MPI_STATUSES_IGNORE);

		c0 = MPI_Wtime();
		update_E_range_omp(1, 1, start_x, N, Ce, Ex.data(), Ey.data(), Ez.data(), Hx.data(), Hy.data(), Hz.data());
		if (ic >= start_x && ic < start_x + local_nx) 
		{
			Ez[idx(ic - start_x + 1, jc, kc, N)] += source_pulse(t, dt);
		}
		t_calc_local += (MPI_Wtime() - c0);
	}

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

	MPI_Gatherv(&Ez[1 * plane], local_nx * plane, MPI_FLOAT, Ez_gathered.data(), recvcounts.data(), displs.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	double t_total = MPI_Wtime() - t_total_start;

	double t_calc_max = 0.0;
	MPI_Reduce(&t_calc_local, &t_calc_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (rank == 0) 
	{
		int total_cores = size * threads_per_proc;
		double t_ovh = (t_total > t_calc_max) ? (t_total - t_calc_max) : 0.0;
		double ovh_pct = (t_ovh / t_total) * 100.0;

		double max_err = 0.0;
		for (size_t m = 0; m < Ez_ref.size(); ++m) 
		{
			double diff = std::fabs(Ez_gathered[m] - Ez_ref[m]);
			if (diff > max_err) max_err = diff;
		}

		double s_calc = t_seq / t_calc_max;
		double e_calc = s_calc / total_cores;
		double s_tot = t_seq / t_total;
		double e_tot = s_tot / total_cores;
		double cells_updated = (double)(N - 2) * (N - 2) * (N - 2) * steps;
		double mcells_s = (cells_updated / t_total) / 1e6;
		double bw_gbs = (cells_updated * 96.0) / (t_total * 1e9);

		std::printf(" 3D FDTD Hybrid MPI + OpenMP | Grid: %d^3 | Steps: %d | Config: %d MPI x %d OMP (%d cores) | T_seq = %.4f s\n", N, steps, size, threads_per_proc, total_cores, t_seq);
		std::printf("| MPIxOMP | T_calc,s | T_ovh,s  | T_tot,s  | Ovh,%%  | S_calc | E_clc | S_tot  | E_tot | MCells/s | GB/s   | L_inf   |\n");
		std::printf("| %2dx%-2d   | %8.4f | %8.4f | %8.4f | %5.1f%% | %6.2f | %5.3f | %6.2f | %5.3f | %8.2f | %6.2f | %.1e |\n", size, threads_per_proc, t_calc_max, t_ovh, t_total, ovh_pct, s_calc, e_calc, s_tot, e_tot, mcells_s, bw_gbs, max_err);
	}

	MPI_Finalize();
	return 0;
}