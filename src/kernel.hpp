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
#define NF 6
#define XDIM 0
#define YDIM 1
#define ZDIM 2
#define COORD_CARTESIAN
#define COORD_CYLINDRICAL
#define COORD_SPHERICAL
#define den_i 0
#define ene_i 1
#define tau_i 2
#define mom_i 3

#define FGAMMA real(7.0/5.0)


#ifdef _DOUBLE
#define DB_REAL DB_DOUBLE
using real = double;
#else
#define DB_REAL DB_FLOAT
using real = float;
#endif

real cuda_hydro_wrapper(real* rho, real* s[NDIM], real* egas, int nx,
		int ny, int nz, real dx, real dy, real dz);
void cuda_exit();

#endif /* KERNEL_HPP_ */
