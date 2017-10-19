# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:15:55 2016

@author: ameresj
"""

from __future__ import absolute_import, print_function
import pyopencl as cl
import pyopencl.array as cla
import pyopencl.clmath as clm
import numpy as np
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel
import timeit
import math as math
import matplotlib.pyplot as plt
# Select the my INTEL GPU using the intel driver
# 
platform = cl.get_platforms()[0]    # Select the first platform [0]
device = platform.get_devices()[1]  # Select the first device on this platform [0]
ctx = cl.Context([device])
clq = cl.CommandQueue(ctx)

print(platform)
print(device)

Np=1e5  # Number of particles
Nf=64    # Number of Fourier modes
dt=0.1
tmax=15

L=2*np.pi/0.5


## Convert some input to integers
Np=np.int(Np)
Nf=np.int(Nf)


dtype=np.float64
dtype2=np.complex128

# Unit mode for particle in Fourier
kx0= (2*np.pi)/L
kx=np.linspace(1,Nf,Nf)*kx0
kx=cla.to_device(clq,kx)

def f0(x,v):
    return np.exp(-0.5*v**2)/np.sqrt(2*np.pi)*(1.0+0.1*np.cos(0.5*x ))
    
def g0(x,v):
    return np.exp(-0.5*v**2)/np.sqrt(2*np.pi)/L
    
prg = cl.Program(ctx, """
  #define PYOPENCL_DEFINE_CDOUBLE
  #include <pyopencl-complex.h>
__kernel void project_pif(
    __global const double *xn, __global const cdouble_t *rhs, __global double *En,
      const int Nf, const double kx0)
{
  int gid = get_global_id(0);
  double sini,cosi;
  En[gid]=0;
  for( int kdx = 0; kdx < Nf; kdx++){
      sini = sincos(xn[gid]*(kdx+1)*kx0, &cosi);  
      En[gid]=En[gid]+2*(rhs[kdx].real*cosi - rhs[kdx].imag*sini);
  }  
}
__kernel void project_pif2(
    __global const double *xn, __global const cdouble_t *rhs, __global double *En,
      const int Nf, const double kx0)
{
  int gid = get_global_id(0);
  cdouble_t psi0, psik;
  psi0.imag = sincos(xn[gid]*kx0, &psi0.real);  
  psik=cdouble_new(1,0);
  En[gid]=0;
  for( int kdx = 0; kdx < Nf; kdx++){

      psik=cdouble_mul(psi0,psik);
      En[gid]=En[gid]+2*(rhs[kdx].real*psik.real - rhs[kdx].imag*psik.imag);
  }  
}

__kernel void accum_pif(__global const double * xn ,__global const double * wn,
                    __global cdouble_t * rhs, 
                          const int Nf, const int Np, const double kx0)
{
  //cdouble_t sum = cdouble_new(1,0);
  double sumi,sumr;
  double sini,cosi;

  int kdx = get_global_id(0); // particle index
  sumi=0; sumr=0;
  for (int pdx=0;pdx<Np;pdx++)
    {
      sini = sincos(xn[pdx]*(kdx+1.0)*kx0, &cosi); 
      sumr+=cosi*wn[pdx];
      sumi+=sini*wn[pdx];
      //sum += a[kdx + m*k] * wn[pdx];
    }
  
   rhs[kdx] = cdouble_new(sumr,-sumi);
}
#define ROW_DIM 0
#define COL_DIM 1
// P threads per row compute 1/P-th of each dot product.
// WORK has P columns and get_local_size(0) rows.
__kernel void accum_pif3(__global const double * xn ,__global const double * wn,
                    __global cdouble_t * rhs, 
		    __local cdouble_t * work,
              const int Nf, const int Np, const double kx0)
{
  double sumi,sumr;
  double sini,cosi;
  
  // Compute partial dot product
  sumi=0; sumr=0;
  for (int pdx=get_global_id(COL_DIM);pdx<Np;pdx+=get_global_size(COL_DIM))
    {
     sini = sincos(xn[pdx]*(get_global_id(ROW_DIM)+1.0)*kx0, &cosi); 
      sumr+=cosi*wn[pdx];
      sumi+=sini*wn[pdx];  
    }

  // Each thread stores its partial sum in WORK
  int rows = get_local_size(ROW_DIM); // rows in group
  int cols = get_local_size(COL_DIM); // initial cols in group
  int ii = get_local_id(ROW_DIM); // local row index in group, 0<=ii<rows
  int jj = get_local_id(COL_DIM); // block index in column, 0<=jj<cols
  
  work[ii+rows*jj] = cdouble_new(sumr,-sumi);
  barrier(CLK_LOCAL_MEM_FENCE); // sync group

  // Reduce sums in log2(cols) steps
  while ( cols > 1 )
    {
      cols >>= 1;
      if (jj < cols){
          work[ii+rows*jj] = cdouble_add( work[ii+rows*jj]    ,work[ii+rows*(jj+cols)]);
      } 
      
      barrier(CLK_LOCAL_MEM_FENCE); // sync group
    }

  // Write final result in Y
  if ( jj == 0 ) rhs[get_global_id(ROW_DIM)] = work[ii];
}

""").build()








accum_mode = ReductionKernel(ctx, np.complex128, neutral="cdouble_new(0,0)",
        reduce_expr="cdouble_add(a,b)",
        map_expr="cdouble_mulr(exp_complex(kx0*xn[i]),wn[i])",
        arguments="""__global const double *xn, __global const double *wn,
                     const double kx0""",
        preamble=""" 
          #define PYOPENCL_DEFINE_CDOUBLE
          #include <pyopencl-complex.h>
          cdouble_t exp_complex( double x )
          {
              cdouble_t y;
              y.imag = sincos(x, &y.real);              
              return y;
          }
        """);
        
# Coefficients for Runge Kutta
rksd=np.array([2/3.0, -2/3.0, 1  ]);
rksc=np.array([ 7/24.0, 3/4.0, -1/24.0]);
 
   
# Allocate particle space on GPU
xn=cla.zeros(clq, (Np,) ,dtype)
vn=cla.zeros(clq, (Np,) ,dtype)
fn=cla.zeros(clq, (Np,) ,dtype)
gn=cla.zeros(clq, (Np,) ,dtype)
Ex=cla.zeros(clq, (Np,) ,dtype)

rhs=cla.zeros(clq, (Nf,),dtype2)



# Sample particles
xn=cla.to_device(clq,np.random.rand(Np,).astype(dtype))
xn=xn*L
vn=cla.to_device(clq,np.random.randn(Np,).astype(dtype))

fn=cla.to_device(clq,f0(xn.get(),vn.get()))
gn=cla.to_device(clq,g0(xn.get(),vn.get()))

num_t=int(math.floor(tmax/dt))


tic=timeit.default_timer()
fieldenergy=np.zeros((num_t,1))

for tdx in range(0, num_t  ):
    wn=fn/gn
    
    for rkdx in range(0,rksd.size):
    
        # Accumulate charge
        #for kdx in range(0,Nf):
            #rhs[kdx]=cla.sum(clm.exp(-1j*xn*(kdx+1.0)*kx0  )*wn )/L/Np
            #rhs[kdx]=accum_mode(xn,wn,-kx0*(kdx+1.0))/L/Np  
        
        #FOR GPU            
#        for kdx in range(0,Nf):
#           accum_mode.__call__( xn,wn,-kx0*(kdx+1.0), queue=clq,out=rhs[kdx])
#        clq.flush()
            
        # FOR CPU
        # One thread per dot product (over particles)
        kernel=prg.accum_pif
        kernel.set_scalar_arg_dtypes([None, None, None,  np.int32,np.int32, np.float64])
        kernel(clq, rhs.shape, None , xn.data, wn.data, rhs.data, Nf,Np, kx0).wait()
        
        rhs=rhs/L/Np 

        #Poisson solve
        rhs=-rhs/(-1j*kx)
        
        if rkdx==0:
            fieldenergy[tdx]=cla.dot(rhs,rhs.conj()).real.get()/2.0
        
        #Ex[:]=0
        #for kdx in range(0,Nf):
        #       Ex=Ex+ 2*(rhs[kdx].real.get()*clm.cos(xn*(kdx+1.0)*kx0) + 
        #            rhs[kdx].imag.get()*clm.sin(xn*(kdx+1.0)*kx0) )
        kernel=prg.project_pif2
        kernel.set_scalar_arg_dtypes([None, None, None, np.int32, np.float64])
        kernel(clq, Ex.shape, None , xn.data, rhs.data, Ex.data,Nf, kx0)
        
        
        vn=vn+dt*rksc[rkdx]*Ex
        xn=xn+dt*rksd[rkdx]*vn
    
    
    
print(sum(fieldenergy))    
    
toc=timeit.default_timer()


print(toc-tic)

#clL=cla.to_device(clq,np.float64(L))

#xn= clm.fmod(xn,clL )
#time=np.linspace(0,tmax,num_t)
#plt.figure()
#plt.semilogy(time,fieldenergy)
#
#
#plt.figure()
#
#plt.scatter(np.mod(xn.get(),L), vn.get(), s=20, c=fn.get(),cmap='viridis',lw = 0)
#
#plt.colorbar()


