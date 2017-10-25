#include "kernel.hpp"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <silo.h>

void write_silo(real* U[NF], const char* filename, int nx, int ny, int nz,
		real dx, real dy, real dz) {
	int sz = nx * ny * nz;
	int sz_coord = (nx + 1) * (ny + 1) * (nz + 1);
	real* coords[NDIM];
	char* coordnames[NDIM];
	int dims[] = { nx, ny, nz };
	int dims_coord[] = { nx + 1, ny + 1, nz + 1 };
	for (int dim = 0; dim != NDIM; ++dim) {
		coords[dim] = new real[sz_coord];
		coordnames[dim] = new char[2];
		coordnames[dim][0] = 'x' + dim;
		coordnames[dim][1] = '\0';
	}
	for (int xi = 0; xi != nx + 1; ++xi) {
		for (int yi = 0; yi != ny + 1; ++yi) {
			for (int zi = 0; zi != nz + 1; ++zi) {
				const int iii = xi + (nx + 1) * (yi + (ny + 1) * zi);
				coords[XDIM][iii] = xi * dx;
				coords[YDIM][iii] = yi * dy;
				coords[ZDIM][iii] = zi * dz;
			}
		}
	}
	auto db = DBCreate(filename, DB_CLOBBER, DB_LOCAL, "CUDA Hydro", DB_PDB);

	DBPutQuadmesh(db, "mesh", coordnames, coords, dims_coord, NDIM, DB_REAL,
			DB_NONCOLLINEAR, NULL);

	const char* names[] = { "rho", "egas", "tau", "sx", "sy", "sz" };

	for (int f = 0; f != NF; ++f) {
		DBPutQuadvar1(db, names[f], "mesh", U[f], dims, NDIM, NULL, 0, DB_REAL,
				DB_ZONECENT, NULL);
	}

	DBClose(db);
	for (int dim = 0; dim != NDIM; ++dim) {
		delete[] coords[dim];
		delete[] coordnames[dim];
	}
}

int main() {

	const int nx = 128;
	const int ny = 128;
	const int nz = 128;

	real* U[NF];
	for (int f = 0; f != NF; ++f) {
		U[f] = new real[nx * ny * nz];
	}

	for (int i = 0; i != nx; ++i) {
		for (int j = 0; j != ny; ++j) {
			for (int k = 0; k != nz; ++k) {
				const int iii = i + nx * j + nx * ny * k;
				for (int f = 0; f != NF; ++f) {
					U[f][iii] = 0.0;
				}
				if (i < nx / 2) {
					U[den_i][iii] = 1.0;
					U[ene_i][iii] = 2.5;
				} else {
					U[den_i][iii] = 0.125;
					U[ene_i][iii] = 0.25;
				}
				U[tau_i][iii] = std::pow(U[den_i][iii], 1.0 / FGAMMA);
			}
		}
	}
	real* s[NDIM] = { U[mom_i + XDIM], U[mom_i + YDIM], U[mom_i + ZDIM] };
	real t = 0.0;

	const auto silo_out = [&](int index) {
		char* ptr;
		if (asprintf(&ptr, "X.%i.silo", index) == 0) {
			assert(false);
			printf("error\n");
			abort();
		}
		write_silo(U, ptr, nx, ny, nz, 1.0 / nx, 1.0 / ny, 1.0 / nz);
		free(ptr);
	};
	silo_out(0);
	for (int ti = 0; ti < 100; ++ti) {
		auto dt = cuda_hydro_wrapper(U[den_i], s, U[ene_i], nx, ny, nz,
				1.0 / nx, 1.0 / ny, 1.0 / nz);
		printf("%i %e %e\n", ti, t, dt);
		t += dt;
		if (ti % 10 == 0) {
			silo_out(ti + 1);
		}
	}
	cuda_exit();
	return EXIT_SUCCESS;
}
