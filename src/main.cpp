#include "kernel.hpp"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

int main() {

	const int nx = 128;
	const int ny = nx;
	const int nz = nx;

	double* U[NF];
	for (int f = 0; f != NF; ++f) {
		U[f] = new double[nx * ny * nz];
	}

	for (int i = 0; i != nx; ++i) {
		for (int j = 0; j != ny; ++j) {
			for (int k = 0; k != nz; ++k) {
				const int iii = i + nx * j + nx * ny * k;
				for (int f = 0; f != NF; ++f) {
					U[f][i] = 0.0;
				}
				if (k < nz / 2) {
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
	double* s[NDIM] = { U[mom_i + XDIM], U[mom_i + YDIM], U[mom_i + ZDIM] };
	double t = 0.0;
	for (int ti = 0; ti < 100; ++ti) {
		auto dt = cuda_hydro_wrapper(U[den_i], s, U[ene_i], nx, ny, nz,
				1.0 / nz);
		char* ptr;
		if (asprintf(&ptr, "X.%i.dat", int(ti)) == 0) {
			assert(false);
			printf("error\n");
			abort();
		}
		FILE* fp = fopen(ptr, "wt");
		for (int i = 0; i != nz; ++i) {
			const int iii = BW + nx * (BW) + nx * ny * i;
			fprintf(fp, "%e %e %e %e %e %e\n", double(i), U[den_i][iii],
					U[mom_i + 0][iii], U[mom_i + 1][iii], U[mom_i + 2][iii],
					U[ene_i][iii]);
		}
		fclose(fp);
		free(ptr);
		printf("%i %e %e\n", ti, t, dt);
	}
	cuda_exit();
	return EXIT_SUCCESS;
}
