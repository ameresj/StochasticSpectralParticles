#!/bin/bash
# Compile 
#gfortran -O3 -o pif_vp.fortran    pif_vp.F90  
gfortran -O3 -mtune=native -march=native -o simple_vp_pif.fortran2    simple_vp_pif.F90  
gfortran -O3 -mtune=native -march=native -fopenmp  -o pif_vp_OMP.fortran pif_vp_OMP.F90
gfortran -O3 -mtune=native -march=native -fopenmp  -o pif_vp_std_OMP.fortran pif_vp_OMP.F90


# Run single core tests
export OMP_NUM_THREADS=1
# echo "gFortran -O3"
# time ./simple_vp_pif.fortran
# echo "gFortran -O3 -march=native"
# time ./simple_vp_pif.fortran2
# echo "gFortran -O3 -march=native -fopenmp -fopenmp-simd"
# time ./simple_vp_pif_OMP.fortran 
# echo "julia"
# 
# time julia -O3 --compile=all --depwarn=no simple_vp_pif.jl


# PERF
gfortran -O3 -mtune=native -march=native -fopenmp -ggdb -o pif_vp_OMP.fortran pif_vp_OMP.F90

perf stat -B -e cache-references,cache-misses,cycles,instructions,branches,faults,migrations pif_vp_OMP.fortran