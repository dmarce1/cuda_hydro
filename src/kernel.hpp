/*
 * kernel.hpp
 *
 *  Created on: Oct 23, 2017
 *      Author: dmarce1
 */

#ifndef KERNEL_HPP_
#define KERNEL_HPP_

#define BW 2
#define NDIM 3
#define NF 5
#define XDIM 0
#define YDIM 1
#define ZDIM 2
#define den_i 0
#define ene_i 1
#define mom_i 2


double cuda_hydro_wrapper(double* rho, double* s[NDIM], double* egas, int nx,
		int ny, int nz, double dx);

#endif /* KERNEL_HPP_ */
