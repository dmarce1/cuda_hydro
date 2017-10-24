#include <limits>
#include <future>
#include "kernel.hpp"

#define NRK 2
#define CFL 0.4

#define INVOKE( func, blocks, threads, ...) \
	func <<< blocks, threads >>> ( __VA_ARGS__ ); \
	cudaStreamSynchronize(0)

__device__
double minmod(double a, double b) {
	return (copysign(0.5, a) + copysign(0.5, b)) * fmin(fabs(a), fabs(b));
}

__global__
void cuda_prep(double* U_base, double* U0_base, double* dU_base) {
	const int nx = blockDim.x;
	const int ny = gridDim.x;
	const int nz = gridDim.y;
	const int xi = threadIdx.x;
	const int yi = blockIdx.x;
	const int zi = blockIdx.y;
	const int nx1 = nx + 2 * BW;
	const int ny1 = ny + 2 * BW;
	const int nz1 = nz + 2 * BW;
	const int sz = nx1 * ny1 * nz1;
	const int idx = (xi + BW) + nx1 * (yi + BW) + (nx1 * ny1) * (zi + BW);
	double* U[NF];
	double* U0[NF];
	double* dU[NF];
	for (int f = 0; f != NF; ++f) {
		U[f] = U_base + f * sz;
		U0[f] = U0_base + f * sz;
		dU[f] = dU_base + f * sz;
	}
	for (int f = 0; f != NF; ++f) {
		dU[f][idx] = 0.0;
		U0[f][idx] = U[f][idx];
	}

}

__global__
void cuda_hydro(double* U_base, double* dU_base, double dx, int dim, int di,
		double* dt_max) {
	int nx, ny, nz;
	int xi, yi, zi;
	int nx1, ny1, nz1;

	switch (dim) {
	case XDIM:
		nx = blockDim.x;
		ny = gridDim.x;
		nz = gridDim.y;
		xi = threadIdx.x;
		yi = blockIdx.x;
		zi = blockIdx.y;
		nx1 = nx + 2 * BW - 1;
		ny1 = ny + 2 * BW;
		nz1 = nz + 2 * BW;
		break;
	case YDIM:
		ny = blockDim.x;
		nx = gridDim.x;
		nz = gridDim.y;
		yi = threadIdx.x;
		xi = blockIdx.x;
		zi = blockIdx.y;
		nx1 = nx + 2 * BW;
		ny1 = ny + 2 * BW - 1;
		nz1 = nz + 2 * BW;
		break;
	case ZDIM:
		nz = blockDim.x;
		nx = gridDim.x;
		ny = gridDim.y;
		zi = threadIdx.x;
		xi = blockIdx.x;
		yi = blockIdx.y;
		nx1 = nx + 2 * BW;
		ny1 = ny + 2 * BW;
		nz1 = nz + 2 * BW - 1;
		break;
	}

	const int idx = (xi + BW) + nx1 * (yi + BW) + (nx1 * ny1) * (zi + BW);
	const int sz = (nx1) * (ny1) * (nz1);
	const double gamma = 5.0 / 3.0;
	double F[NF];
	double* U[NF];
	double* dU[NF];
	for (int f = 0; f != NF; ++f) {
		U[f] = U_base + f * sz;
		dU[f] = dU_base + f * sz;
	}
	const int mom_dim = mom_i + dim;
	if (threadIdx.x == 0) {
		for (int f = 0; f != NF; ++f) {
			U[f][idx - 2 * di] = U[f][idx - di] = U[f][idx];
		}
		U[mom_dim][idx - 2 * di] = U[mom_dim][idx - di] = fmax(0.0,
				U[mom_dim][idx]);
	}
	__syncthreads();
	if (threadIdx.x == blockDim.x - 2) {
		for (int f = 0; f != NF; ++f) {
			U[f][idx + 2 * di] = U[f][idx + di] = U[f][idx];
		}
		U[mom_dim][idx + 2 * di] = U[mom_dim][idx + di] = fmin(0.0,
				U[mom_dim][idx]);
	}
	__syncthreads();

	double up1, um1;
	double up2, um2;
	double slm, slp;
	double UR[NF], UL[NF];
	double ar, al;
	double pl, pr;
	double vl, vr;
	double cl, cr;
	double ekr, ekl;
	double eir, eil;
	double a;
	for (int f = 0; f != NF; ++f) {
		um2 = U[f][idx - 2 * di];
		um1 = U[f][idx - di];
		up1 = U[f][idx];
		up2 = U[f][idx + di];
		slm = minmod(um1 - um2, up1 - um1);
		slp = minmod(up1 - um1, up2 - up1);
		UL[f] = um1 + 0.5 * slm;
		UR[f] = up1 - 0.5 * slp;
	}
	ekr = ekl = 0.0;
	for (int d = 0; d != NDIM; ++d) {
		ekl += UL[mom_i + d] * UL[mom_i + d];
		ekr += UR[mom_i + d] * UR[mom_i + d];
	}
	ekl *= 0.5 / UL[den_i];
	ekr *= 0.5 / UR[den_i];
	eil = UL[ene_i] - ekl;
	eir = UR[ene_i] - ekr;
	pl = (gamma - 1.0) * eil;
	pr = (gamma - 1.0) * eir;
	cl = sqrt(gamma * pl / UL[den_i]);
	cr = sqrt(gamma * pr / UR[den_i]);
	vl = UL[mom_dim] / UL[den_i];
	vr = UR[mom_dim] / UR[den_i];
	al = fabs(vl) + cl;
	ar = fabs(vr) + cr;
	a = fmax(al, ar);
	for (int f = 0; f != NF; ++f) {
		F[f] = -a * (UR[f] - UL[f]);
		F[f] += vr * UR[f] + vl * UL[f];
	}
	F[mom_dim] += pl + pr;
	F[ene_i] += vl * pl + vr * pr;
	for (int f = 0; f != NF; ++f) {
		F[f] *= 0.5;
	}
	for (int f = 0; f != NF; ++f) {
		atomicAdd(&(dU[f][idx]), F[f] / dx);
		atomicAdd(&(dU[f][idx - di]), -F[f] / dx);
	}
	if (dt_max != nullptr) {
		const int tid = threadIdx.x;
		const int max_id = blockDim.x;
		const int D = gridDim.x * gridDim.y;
		const int offset = blockIdx.x + gridDim.x * blockIdx.y;
		const int myid = D * tid + offset;
		const int topid = 1 << (31 - __clz(max_id));
		dt_max[myid] = dx / a;
		for (int num = topid; num > 0; num >>= 2) {
			if ((tid < num) && (tid + num < topid)) {
				const int o_id = D * (tid + num) + offset;
				dt_max[myid] = fmin(dt_max[myid], dt_max[o_id]);
			}
			__syncthreads();
		}
	}

}

__global__
void cuda_advance(double* U_base, double* U0_base, double* dU_base, double dt,
		double beta) {
	const int nx1 = blockDim.x + 2 * BW;
	const int ny1 = gridDim.x + 2 * BW;
	const int nz1 = gridDim.y + 2 * BW;
	const int xi = threadIdx.x;
	const int yi = blockIdx.x;
	const int zi = blockIdx.y;
	const int idx = (xi + BW) + nx1 * (yi + BW) + (nx1 * ny1) * (zi + BW);
	const int sz = (nx1) * (ny1) * (nz1);
	double* U[NF];
	double* U0[NF];
	double* dU[NF];
	for (int f = 0; f != NF; ++f) {
		U[f] = U_base + f * sz;
		U0[f] = U0_base + f * sz;
		dU[f] = dU_base + f * sz;
	}
	__syncthreads();
	for (int f = 0; f != NF; ++f) {
		U[f][idx] += beta * dt * dU[f][idx]
				+ (U0[f][idx] - U[f][idx]) * (1.0 - beta);
		dU[f][idx] = 0.0;
	}

}

double cuda_hydro_wrapper(double* rho, double* s[NDIM], double* egas, int nx,
		int ny, int nz, double dx) {
	static bool first_call = true;
	const int sz = nx * ny * nz;

	static double* U;
	static double* U0;
	static double* dU;
	static double* dt_max[NDIM];
	static double* local_dt_max[NDIM];

	static dim3 blocks[NDIM];
	static dim3 threads[NDIM];
	static dim3 blocks0(ny - 2 * BW, nz - 2 * BW);
	static dim3 threads0(nx - 2 * BW);

	double dt;

	if (first_call) {
		blocks[XDIM] = dim3(ny - 2 * BW, nz - 2 * BW);
		blocks[YDIM] = dim3(nx - 2 * BW, nz - 2 * BW);
		blocks[ZDIM] = dim3(nx - 2 * BW, ny - 2 * BW);
		threads[XDIM] = dim3(nx - 2 * BW + 1);
		threads[YDIM] = dim3(ny - 2 * BW + 1);
		threads[ZDIM] = dim3(nz - 2 * BW + 1);
		for (int dim = 0; dim != NDIM; ++dim) {
			const int this_sz = blocks[dim].x * blocks[dim].y * threads[dim].x;
			cudaMalloc(&(dt_max[dim]), this_sz * sizeof(double));
			local_dt_max[dim] = new double[blocks[dim].x * blocks[dim].y];
		}
		cudaMalloc(&U, NF * sz * sizeof(double));
		cudaMalloc(&U0, NF * sz * sizeof(double));
		cudaMalloc(&dU, NF * sz * sizeof(double));
		first_call = false;
	}

	cudaMemcpy(U + sz * den_i, rho, sz * sizeof(double),
			cudaMemcpyHostToDevice);
	cudaMemcpy(U + sz * ene_i, egas, sz * sizeof(double),
			cudaMemcpyHostToDevice);
	for (int d = 0; d != NDIM; ++d) {
		cudaMemcpy(U + sz * (mom_i + d), s[d], sz * sizeof(double),
				cudaMemcpyHostToDevice);
	}

	const int di[NDIM] = { 1, nx, nx * ny };

	INVOKE(cuda_prep, blocks0, threads0, U, U0, dU);

	std::future<double> dt_fut[NDIM];

	dt = std::numeric_limits<double>::max();

	for (int rk = 0; rk < NRK; ++rk) {
		const double beta = rk == 0 ? 1.0 : 0.5;

		for (int dim = 0; dim != NDIM; ++dim) {
			dt_fut[dim] =
					std::async(std::launch::async,
							[=]() {
								INVOKE(cuda_hydro, (blocks[dim]),(threads[dim]),U, dU, dx, dim, (di[dim]), (rk == 0 ? dt_max[dim] : nullptr));
								double dt = std::numeric_limits<double>::max();
								if (rk == 0) {
									const int this_sz = blocks[dim].x * blocks[dim].y;
									cudaMemcpy(local_dt_max[dim], dt_max[dim], this_sz * sizeof(double),
											cudaMemcpyDeviceToHost);
									for (int b = 0; b < this_sz; ++b) {
										dt = std::min(dt, CFL * local_dt_max[dim][b]);
									}
								}
								return dt;
							});
		}

		for (int dim = 0; dim != NDIM; ++dim) {
			const double this_dt = dt_fut[dim].get();
			if (rk == 0) {
				dt = std::min(dt, this_dt);
			}
		}
		/**/INVOKE(cuda_advance, blocks0, threads0, U, U0, dU, dt, beta);

		/**/}

	cudaMemcpy(rho, U + sz * den_i, sz * sizeof(double),
			cudaMemcpyDeviceToHost);
	cudaMemcpy(egas, U + sz * ene_i, sz * sizeof(double),
			cudaMemcpyDeviceToHost);
	for (int d = 0; d != NDIM; ++d) {
		cudaMemcpy(s[d], U + sz * (mom_i + d), sz * sizeof(double),
				cudaMemcpyDeviceToHost);
	}

	return dt;
}
