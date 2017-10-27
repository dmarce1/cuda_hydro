#include <limits>
#include <future>
#include "kernel.hpp"

#ifdef _DOUBLE
#pragma message( "Compiling with double precision.")
#if  _DU_DOUBLE
using du_real = double;
#else
using du_real = float;
#pragma message( "Compiling with single precision accumulator.")
#endif
#else
#pragma message( "Compiling with single precision.")
using du_real = float;
#endif

#define NRK 2
#define CFL real(0.4)
#define DE_SWITCH_1 real(0.001)
#define DE_SWITCH_2 real(0.01)
#define THETA real(1.3)

#define INVOKE2( func, blocks, threads, ...) \
	func <<< blocks, threads >>> ( __VA_ARGS__ ); \
	cudaStreamSynchronize(0)

#define INVOKE3( func, blocks, threads, size, ...) \
	func <<< blocks, threads, size >>> ( __VA_ARGS__ ); \
	cudaStreamSynchronize(0)

#ifdef CARTESIAN
__device__
inline real h1(real x, real y, real z) {
	return real(1);
}
__device__
inline real h2(real x, real y, real z) {
	return real(1);
}
__device__
inline real h3(real x, real y, real z) {
	return real(1);
}
__device__
inline real hhh(real x, real y, real z) {
	return real(1);
}
#endif

#ifdef CYLINDRICAL
__device__
inline real h1(real x, real y, real z) {
	return real(1);
}
__device__
inline real h2(real x, real y, real z) {
	return x;
}
__device__
inline real h3(real x, real y, real z) {
	return real(1);
}
__device__
inline real hhh(real x, real y, real z) {
	return x;
}
#endif

#ifdef SPHERICAL
__device__
inline real h1(real x, real y, real z) {
	return real(1);
}
__device__
inline real h2(real x, real y, real z) {
	return x;
}
__device__
inline real h3(real x, real y, real z) {
	return real(1);
}
__device__
inline real hhh(real x, real y, real z) {
	return x;
}
#endif

__device__ static double* devXf[NDIM];
__device__ static double* devXc[NDIM];
static double* hostXf[NDIM];
static double* hostXc[NDIM];

__global__
void set_device_ptrs(double* xc, double* yc, double* zc, double* xf, double* yf,
		double* zf) {
	devXf[XDIM] = xf;
	devXf[YDIM] = yf;
	devXf[ZDIM] = zf;
	devXc[XDIM] = xc;
	devXc[YDIM] = yc;
	devXc[ZDIM] = zc;
}

void create_coordinates(real dx, real dy, real dz, int nx, int ny, int nz) {
	double* dXc[NDIM];
	double* dXf[NDIM];
	const int sz = nx * ny * nz;
	for (int d = 0; d != NDIM; ++d) {
		hostXf[d] = new double[sz];
		hostXc[d] = new double[sz];
		cudaMalloc(&(dXf[d]), sz * sizeof(real));
		cudaMalloc(&(dXc[d]), sz * sizeof(real));
	}
	real x0, y0, z0;
#ifdef CARTESIAN
	x0 = y0 = z0 = 0.0;
#endif
#ifdef CYLINDRICAL
	x0 = y0 = 0.0;
	z0 = (nz - 2 * BW + 1) * dz / 2.0;
#endif
#ifdef SPHERICAL
	x0 = y0 = 0.0;
	z0 = -1.0;
#endif
	for (int i = 0; i != nx; ++i) {
		for (int j = 0; j != ny; ++j) {
			for (int k = 0; k != nz; ++k) {
				const int iii = i + nx * (j + ny * k);
				hostXf[XDIM][iii] = (i - BW) * dx + x0;
				hostXf[YDIM][iii] = (j - BW) * dy + y0;
				hostXf[ZDIM][iii] = (k - BW) * dz + z0;
				hostXc[XDIM][iii] = (i - BW + 0.5) * dx + x0;
				hostXc[YDIM][iii] = (j - BW + 0.5) * dy + y0;
				hostXc[ZDIM][iii] = (k - BW + 0.5) * dz + z0;
			}
		}
	}
	for (int d = 0; d != NDIM; ++d) {
		cudaMemcpy(&(dXf[d]), hostXf[d], sz * sizeof(real),
				cudaMemcpyHostToDevice);
		cudaMemcpy(&(dXc[d]), hostXc[d], sz * sizeof(real),
				cudaMemcpyHostToDevice);
	}
	INVOKE2(set_device_ptrs, 1, 1,
			(dXc[XDIM]), (dXc[YDIM]), (dXc[ZDIM]), (dXf[XDIM]), (dXf[YDIM]), (dXf[ZDIM]));
}

__device__
real h(int dim, real x, real y, real z) {
	typedef real (*func_type)(real, real, real);
	static const func_type func[NDIM] = { h1, h2, h3 };
	return func[dim](x, y, z);
}

__device__
inline real minmod(real a, real b) {
	return (copysign(0.5, a) + copysign(0.5, b)) * fmin(fabs(a), fabs(b));
}

__device__
inline real minmod_theta(real a, real b) {
	return minmod(real(THETA) * minmod(a, b), real(0.5) * (a + b));
}

void cuda_exit() {
	cudaDeviceReset();
}

__global__
void cuda_prep(real* U_base, real* U0_base, du_real* dU_base) {
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
	real * U[NF];
	real * U0[NF];
	du_real* dU[NF];
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
void cuda_flux(real* U_base, du_real* dU_base, real dx, int dim, int di,
		real* avisc) {

	int nx, ny, nz;
	int xi, yi, zi;
	int nx1, ny1, nz1;

	__shared__
	extern real shared_real[];

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
		/*case ZDIM:*/
	default:
		nz = blockDim.x;
		nx = gridDim.x;
		ny = gridDim.y;
		zi = threadIdx.x;
		xi = blockIdx.x;
		yi = blockIdx.y;
		nx1 = nx + 2 * BW;
		ny1 = ny + 2 * BW;
		nz1 = nz + 2 * BW - 1;
	}

	const int idx = (xi + BW) + nx1 * (yi + BW) + (nx1 * ny1) * (zi + BW);
	const int sz = (nx1) * (ny1) * (nz1);
	real F[NF];
	real * U[NF];
	du_real* dU[NF];
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
	} else if (threadIdx.x == blockDim.x - 2) {
		for (int f = 0; f != NF; ++f) {
			U[f][idx + 2 * di] = U[f][idx + di] = U[f][idx];
		}
		U[mom_dim][idx + 2 * di] = U[mom_dim][idx + di] = fmin(0.0,
				U[mom_dim][idx]);
	}
	__syncthreads();

	real dxinv = real(1.0) / dx;
	real slm, slp;
	real vm2[NF], vm1[NF], vp1[NF], vp2[NF];
	real UR[NF], UL[NF];
	real ar, al;
	real pl, pr;
	real vl, vr;
	real cl, cr;
	real ekr, ekl;
	real eir, eil;
	real a;
	const int im2 = idx - 2 * di;
	const int im1 = idx - di;
	const int ip1 = idx;
	const int ip2 = idx + di;
	vm2[den_i] = U[den_i][im2];
	vm1[den_i] = U[den_i][im1];
	vp1[den_i] = U[den_i][ip1];
	vp2[den_i] = U[den_i][ip2];
	const real rho_m2_inv = 1.0 / U[den_i][im2];
	const real rho_m1_inv = 1.0 / U[den_i][im1];
	const real rho_p1_inv = 1.0 / U[den_i][ip1];
	const real rho_p2_inv = 1.0 / U[den_i][ip2];
	for (int f = 1; f < NF; ++f) {
		vm2[f] = U[f][im2] * rho_m2_inv;
		vm1[f] = U[f][im1] * rho_m1_inv;
		vp1[f] = U[f][ip1] * rho_p1_inv;
		vp2[f] = U[f][ip2] * rho_p2_inv;
	}
	for (int f = 0; f != NF; ++f) {
		slm = minmod_theta(vm1[f] - vm2[f], vp1[f] - vm1[f]);
		slp = minmod_theta(vp1[f] - vm1[f], vp2[f] - vp1[f]);
		UL[f] = vm1[f] + 0.5 * slm;
		UR[f] = vp1[f] - 0.5 * slp;
	}
	for (int f = 1; f < NF; ++f) {
		UL[f] *= UL[den_i];
		UR[f] *= UR[den_i];
	}
	ekr = ekl = 0.0;
	for (int d = 0; d != NDIM; ++d) {
		ekl += UL[mom_i + d] * UL[mom_i + d];
		ekr += UR[mom_i + d] * UR[mom_i + d];
	}
	const real rhoLinv = real(1.0) / UL[den_i];
	const real rhoRinv = real(1.0) / UR[den_i];
	ekl *= 0.5 * rhoLinv;
	ekr *= 0.5 * rhoRinv;
	const real etl = UL[ene_i];
	const real etr = UR[ene_i];
	eil = etl - ekl;
	eir = etr - ekr;
	if (eil < etl * DE_SWITCH_1) {
		eil = pow(UL[tau_i], FGAMMA);
	}
	if (eir < etr * DE_SWITCH_1) {
		eir = pow(UR[tau_i], FGAMMA);
	}
	pl = (FGAMMA - 1.0) * eil;
	pr = (FGAMMA - 1.0) * eir;
	cl = sqrt(FGAMMA * pl * rhoLinv);
	cr = sqrt(FGAMMA * pr * rhoRinv);
	vl = UL[mom_dim] * rhoLinv;
	vr = UR[mom_dim] * rhoRinv;
	al = fabs(vl) + cl;
	ar = fabs(vr) + cr;
	a = fmax(al, ar);
	for (int f = 0; f != NF; ++f) {
		F[f] = vr * UR[f] + vl * UL[f] - a * (UR[f] - UL[f]);
	}
	F[mom_dim] += pl + pr;
	F[ene_i] += vl * pl + vr * pr;
	for (int f = 0; f != NF; ++f) {
		F[f] *= 0.5;
	}
	const real x = devXf[XDIM][idx];
	const real y = devXf[YDIM][idx];
	const real z = devXf[ZDIM][idx];
	const real xcp = devXc[XDIM][idx];
	const real ycp = devXc[YDIM][idx];
	const real zcp = devXc[ZDIM][idx];
	const real xcm = devXc[XDIM][idx - di];
	const real ycm = devXc[YDIM][idx - di];
	const real zcm = devXc[ZDIM][idx - di];
	const real h3 = hhh(x, y, z);
	const real hjinv = real(1) / h(dim, x, y, z);
	const real hjinv2 = hjinv * hjinv;
	const real h3pinv = real(1) / hhh(xcp, ycp, zcp);
	const real h3minv = real(1) / hhh(xcm, ycm, zcm);
	const real factorp = dxinv * h3 * hjinv2 * h3pinv;
	const real factorm = dxinv * h3 * hjinv2 * h3minv;
	for (int f = 0; f != NF; ++f) {
		atomicAdd(&(dU[f][idx]), du_real(F[f] * factorp));
		atomicAdd(&(dU[f][idx - di]), -du_real(F[f] * factorm));
	}
	if (avisc != nullptr) {
		int tid = threadIdx.x
				+ blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
		avisc[tid] = a;
	}

}

__global__
void cuda_advance(real* U_base, real* U0_base, du_real* dU_base, real dt,
		real beta) {
	const int nx1 = blockDim.x + 2 * BW;
	const int ny1 = gridDim.x + 2 * BW;
	const int nz1 = gridDim.y + 2 * BW;
	const int xi = threadIdx.x;
	const int yi = blockIdx.x;
	const int zi = blockIdx.y;
	const int idx = (xi + BW) + nx1 * (yi + BW) + (nx1 * ny1) * (zi + BW);
	const int sz = (nx1) * (ny1) * (nz1);
	real * U[NF];
	real * U0[NF];
	du_real* dU[NF];
	for (int f = 0; f != NF; ++f) {
		U[f] = U_base + f * sz;
		U0[f] = U0_base + f * sz;
		dU[f] = dU_base + f * sz;
	}
	for (int f = 0; f != NF; ++f) {
		U[f][idx] += beta * dt * real(dU[f][idx])
				+ (U0[f][idx] - U[f][idx]) * (1.0 - beta);
		dU[f][idx] = du_real(0.0);
	}
	real ek = 0.0;
	const real rhoinv = real(1.0) / U[den_i][idx];
	for (int dim = 0; dim != NDIM; ++dim) {
		ek += U[mom_i + dim][idx] * U[mom_i + dim][idx];
	}
	ek *= 0.5 * rhoinv;
	const real etot = U[ene_i][idx];
	const real ei = etot - ek;
	if (ei > etot * DE_SWITCH_2) {
		U[tau_i][idx] = pow(ei, 1.0 / FGAMMA);
	}
}

__global__
void cuda_source(real* U_base, du_real* dU_base) {
	const int nx1 = blockDim.x + 2 * BW;
	const int ny1 = gridDim.x + 2 * BW;
	const int nz1 = gridDim.y + 2 * BW;
	const int xi = threadIdx.x;
	const int yi = blockIdx.x;
	const int zi = blockIdx.y;
	const int idx = (xi + BW) + nx1 * (yi + BW) + (nx1 * ny1) * (zi + BW);
	const int sz = (nx1) * (ny1) * (nz1);
	real * U[NF];
	du_real* dU[NF];
	for (int f = 0; f != NF; ++f) {
		U[f] = U_base + f * sz;
		dU[f] = dU_base + f * sz;
	}
#ifdef CARTESIAN
#endif
#ifdef CYLINDRICAL
	real ek = 0.0;
	for (int d = 0; d != NDIM; ++d) {
		ek += U[mom_i + d][idx] * U[mom_i + d][idx];
	}
	const real rhoinv = real(1.0) / U[den_i][idx];
	ek *= 0.5 * rhoinv;
	const real et = U[ene_i][idx];
	real ei = et - ek;
	if (ei < et * DE_SWITCH_1) {
		ei = pow(U[tau_i][idx], FGAMMA);
	}
	real p = (FGAMMA - 1.0) * ei;
	const real Rinv = 1.0 / devXc[XDIM][idx];
	const real R3inv = Rinv * Rinv * Rinv;
	dU[mom_i + XDIM][idx] += p * Rinv;
	dU[mom_i + YDIM][idx] += U[den_i][idx] * U[mom_i + YDIM][idx]
			* U[mom_i + YDIM][idx] * R3inv;
#endif
#ifdef SPHERICAL
#endif

}

real cuda_hydro_wrapper(real* rho, real* s[NDIM], real* egas, int nx, int ny,
		int nz, real dx, real dy, real dz) {
	static bool first_call = true;
	const int sz = nx * ny * nz;

	static real* U;
	static real* U0;
	static du_real* dU;
	static real* avisc[NDIM];
	static real* local_avisc[NDIM];
	static real dX[] = { dx, dy, dz };

	static dim3 blocks[NDIM];
	static dim3 threads[NDIM];
	static dim3 blocks0(ny - 2 * BW, nz - 2 * BW);
	static dim3 threads0(nx - 2 * BW);

	real dt;

	if (first_call) {
		create_coordinates(dx, dy, dz, nx, ny, nz);
		blocks[XDIM] = dim3(ny - 2 * BW, nz - 2 * BW);
		blocks[YDIM] = dim3(nx - 2 * BW, nz - 2 * BW);
		blocks[ZDIM] = dim3(nx - 2 * BW, ny - 2 * BW);
		threads[XDIM] = dim3(nx - 2 * BW + 1);
		threads[YDIM] = dim3(ny - 2 * BW + 1);
		threads[ZDIM] = dim3(nz - 2 * BW + 1);
		for (int dim = 0; dim != NDIM; ++dim) {
			const int this_sz = blocks[dim].x * blocks[dim].y * threads[dim].x;
			cudaMalloc(&(avisc[dim]), this_sz * sizeof(real));
			local_avisc[dim] = new real[this_sz];
		}
		cudaMalloc(&U, NF * sz * sizeof(real));
		cudaMalloc(&U0, NF * sz * sizeof(real));
		cudaMalloc(&dU, NF * sz * sizeof(du_real));
		first_call = false;
	}

	cudaMemcpy(U + sz * den_i, rho, sz * sizeof(real), cudaMemcpyHostToDevice);
	cudaMemcpy(U + sz * ene_i, egas, sz * sizeof(real), cudaMemcpyHostToDevice);
	for (int d = 0; d != NDIM; ++d) {
		cudaMemcpy(U + sz * (mom_i + d), s[d], sz * sizeof(real),
				cudaMemcpyHostToDevice);
	}

	const int di[NDIM] = { 1, nx, nx * ny };

	INVOKE2(cuda_prep, blocks0, threads0, U, U0, dU);

	std::future < real > dt_fut[NDIM];

	dt = std::numeric_limits < real > ::max();

	for (int rk = 0; rk < NRK; ++rk) {
		const real beta = rk == 0 ? 1.0 : 0.5;

		for (int dim = 0; dim != NDIM; ++dim) {
			dt_fut[dim] =
					std::async(std::launch::async,
							[=]() {
								INVOKE2(cuda_flux, (blocks[dim]),(threads[dim]),U, dU, dX[dim], dim, (di[dim]), (rk == 0 ? avisc[dim] : nullptr));
								real dt = std::numeric_limits<real>::max();
								if (rk == 0) {
									const int this_sz = blocks[dim].x * blocks[dim].y * threads[dim].x;
									cudaMemcpy(local_avisc[dim], avisc[dim], this_sz * sizeof(real),
											cudaMemcpyDeviceToHost);
									for (int b = 0; b < this_sz; ++b) {
										dt = std::min(dt, CFL * dX[dim] / local_avisc[dim][b]);
									}
								}
								return dt;
							});
		}

		for (int dim = 0; dim != NDIM; ++dim) {
			const real this_dt = dt_fut[dim].get();
			if (rk == 0) {
				dt = std::min(dt, this_dt);
			}
		}
		/**/INVOKE2(cuda_advance, blocks0, threads0, U, U0, dU, dt, beta);

		/**/}

	cudaMemcpy(rho, U + sz * den_i, sz * sizeof(real), cudaMemcpyDeviceToHost);
	cudaMemcpy(egas, U + sz * ene_i, sz * sizeof(real), cudaMemcpyDeviceToHost);
	for (int d = 0; d != NDIM; ++d) {
		cudaMemcpy(s[d], U + sz * (mom_i + d), sz * sizeof(real),
				cudaMemcpyDeviceToHost);
	}

	return dt;
}
