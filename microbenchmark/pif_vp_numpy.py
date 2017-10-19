# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:15:55 2016

@author: ameresj
"""

from __future__ import absolute_import, print_function
import numpy as np
import timeit
import math as math
import matplotlib.pyplot as plt
# Select the my INTEL GPU using the intel driver
# 


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


def f0(x,v):
    return np.exp(-0.5*v**2)/np.sqrt(2*np.pi)*(1.0+0.1*np.cos(0.5*x ))
    
def g0(x,v):
    return np.exp(-0.5*v**2)/np.sqrt(2*np.pi)/L

        
# Coefficients for Runge Kutta
rksd=np.array([2/3.0, -2/3.0, 1  ]);
rksc=np.array([ 7/24.0, 3/4.0, -1/24.0]);
 
   
# Allocate particle space on GPU
xn=np.zeros((Np,) ,dtype)
vn=np.zeros((Np,) ,dtype)
fn=np.zeros((Np,) ,dtype)
gn=np.zeros((Np,) ,dtype)
Ex=np.zeros((Np,) ,dtype)

rhs=np.zeros((Nf,),dtype2)



# Sample particles
xn=np.random.rand(Np,).astype(dtype)
xn=xn*L
vn=np.random.randn(Np,).astype(dtype)

fn=f0(xn,vn)
gn=g0(xn,vn)
wn=fn/gn

num_t=int(math.floor(tmax/dt))


tic=timeit.default_timer()
fieldenergy=np.zeros((num_t,1))

for tdx in range(0, num_t  ):
    
    for rkdx in range(0,rksd.size):
    
        # Accumulate charge
        for kdx in range(0,Nf):
            rhs[kdx]=np.sum(np.exp(-1j*xn*(kdx+1.0)*kx0  )*wn )/L/Np
            
        
        #Poisson solve
        rhs=-rhs/(-1j*kx)
        
        if rkdx==0:
            fieldenergy[tdx]=np.dot(rhs,rhs.conj()).real/2.0
        
        Ex[:]=0
        for kdx in range(0,Nf):
               psin=np.exp(1j*xn*(kdx+1.0)*kx0)
               vn=vn+dt*rksc[rkdx]*2*(rhs[kdx].real*psin.real - rhs[kdx].imag*psin.imag )
        
        
        xn=xn+dt*rksd[rkdx]*vn
    
    
toc=timeit.default_timer()

print(toc-tic)



