# using Devectorize
# import MPI
using ParallelAccelerator
# using PyPlot



Np=Int(1e5)
Nf=Int(64) # Number of Fourier modes
k0=0.5
dt=0.1
tmax=15

L=2*pi/k0


function f(x,v)
  return exp(-0.5.*v.^2)/sqrt(2*pi).*(1+0.1.*cos(k0*x))
end

# MPI.Init()
#
# comm = MPI.COMM_WORLD
# MPI.Barrier(comm)
# mpi_rank = MPI.Comm_rank(comm)
# mpi_size = MPI.Comm_size(comm)
# print(mpi_size)

# Np_loc=Int(floor(Np/mpi_size))
Np_loc=Np

#Seed the random generator
#We would like to skip ahead values
# srand( (Np_loc*4)*(mpi_rank+1) )

xk=rand(Np_loc,1)*L
vk=randn(Np_loc,1)
wk=f(xk,vk)*L

function  rhs_particle(xk::Array{Float64,2},vk::Array{Float64,2},
                wk::Array{Float64,2},Nf::Int,kx0::Float64)
  rho=Array{Complex128}(Nf)
  @inbounds for kdx=1:Nf
    rho[kdx]=sum(exp(-im*kx0*kdx*xk).*wk)
  end
  return rho
end

function  rhs_particle2(xk::Array{Float64,2}, vk::Array{Float64,2},
                wk::Array{Float64,2}, Nf::Int, kx0::Float64)
    Np=length(xk)
    rho=zeros(Complex128,Nf)
    @inbounds for pdx=1:Np
    @simd for kdx=1:Nf
         @inbounds rho[kdx]+=exp(-im*kx0*xk[pdx]*kdx)*wk[pdx]
    end
    end
  return rho
end
function  rhs_particle4(xk::Array{Float64,2}, vk::Array{Float64,2},
                wk::Array{Float64,2}, Nf::Int, kx0::Float64)
    Np=length(xk)
    rho=zeros(Complex128,Nf)
    @inbounds for pdx=1:Np
    @simd for kdx=1:Nf
        arg=kx0*xk[pdx]*kdx
         @inbounds rho[kdx]+=(cos(arg)-
                                im*sin(arg))*wk[pdx]
    end
    end
  return rho
end
function  rhs_particle5{T}(xk::Array{T,2},
         wk::Array{T,2}, Nf::Int, kx0::T)
    Np=length(xk)
    rho=zeros(Complex{T},Nf)
    @inbounds for pdx=1:Np
      @inbounds arg=kx0*xk[pdx]
      psin1=(cos(arg)- im*sin(arg))
      psin=1.0+im*0.0
      for kdx=1:Nf
         psin=psin*psin1
         @inbounds rho[kdx]+=psin*wk[pdx]
       end
    end
  return rho
end

function  push_particle(xk::Array{Float64,2}, vk::Array{Float64,2},
                  E::Array{Complex128,2}, kx0::Float64, dt::Float64)

    Np=length(xk)
    Nf=length(E)
     for pdx=1:Np
        @simd for kdx=1:Nf
         @inbounds psik=exp(im*kx0*kdx*xk[pdx])
         @inbounds vk[pdx]+=dt*2*(real(psik)*real(E[kdx])-imag(psik).*imag(E[kdx]))
      end
    end
    return vk
end

function  push_particle2(xk::Array{Float64,2}, vk::Array{Float64,2},
                  E::Array{Complex128,2}, kx0::Float64, dt::Float64)

    Np=length(xk)
    Nf=length(E)
     @inbounds for pdx=1:Np
        @simd for kdx=1:Nf
        @inbounds arg=kx0*xk[pdx]*kdx
         @inbounds vk[pdx]+=dt*2*(cos(arg)*real(E[kdx])-sin(arg)*imag(E[kdx]))
      end
    end
    return vk
end

function  push_particle5(xk::Array{Float64,2}, vk::Array{Float64,2},
                  E::Array{Complex128,2}, kx0::Float64, dt::Float64)

    Np=length(xk)
    Nf=length(E)
     @inbounds @simd for pdx=1:Np
      @inbounds arg=kx0*xk[pdx]
      psin1=(cos(arg)+ im*sin(arg))
      psin=1.0+im*0.0
      for kdx=1:Nf
          psin=psin*psin1
         @inbounds vk[pdx]+=dt*2*(real(psin)*real(E[kdx])-imag(psin)*imag(E[kdx]))
      end
    end
    return vk
end


function  rhs_particle3(xk::Array{Float64,2},vk::Array{Float64,2},
                wk::Array{Float64,2},Nf::Int,kx0::Float64)
  Np=length(xk)
  rho=Array(Complex128,Nf)
  rho = @parallel (+) for pdx=1:Np
    rhon=Array{Complex128}(Nf)
    @simd  for kdx=1:Nf
         @inbounds rhon[kdx]=exp(-im*kx0*kdx*xk[pdx])*wk[pdx]
      end
    rhon
  end
  return rho
end

# code_llvm(rhs_particle4, (Array{Float64,2},
#    Array{Float64,2},Array{Float64,2},Int,Float64) )

# @profile rho=rhs_particle(xk,vk,wk,Nf,L/(2*pi))
# @time rho=rhs_particle3(xk,vk,wk,Nf,L/(2*pi))
# @time rho=rhs_particle2(xk,vk,wk,Nf,L/(2*pi))
# @time rho=rhs_particle4(xk,vk,wk,Nf,L/(2*pi))
# @time rho=rhs_particle4(xk,vk,wk,Nf,L/(2*pi))
#@time rho=rhs_particle5(xk,wk,Nf,L/(2*pi))
# rho2=zeros(Complex128,Nf)
#  @inbounds for kdx=1:Nf
#   rho2[kdx]=sum(exp(-im*2*pi/L*kdx*xk).*wk)
# end

# Coefficients for Runge Kutta
const rksd=[2/3.0, -2/3.0, 1  ]
const rksc=[ 7/24.0, 3/4.0, -1/24.0]

kx=(1:Nf).''*2*pi/L

Nt=Int(ceil(tmax/dt))
fieldenergy=zeros(Float64,Nt,1)
tic()
for tdx=1:Nt

  @inbounds for rkdx=1:length(rksd)

    rho=rhs_particle5(xk,wk,Nf,(2*pi)/L)
    # rho=zeros(Complex128,Nf)
    # @inbounds for kdx=1:Nf
    #   rho[kdx]=sum(exp(-im*2*pi/L*kdx*xk).*wk)
    # end

    #Allreduce over a vector
    # rho=MPI.allreduce(rho, MPI.SUM, comm)

    rho=rho/L/Np

    E=-rho./(-im*kx)
    if rkdx==1
      fieldenergy[tdx]=(real(E'*E)*L*0.5*2)[1]
    end

    # @inbounds for kdx=1:Nf
    #   psik=exp(im*2*pi/L*kdx*xk)
    #   vk+=dt*rksd[rkdx]*2*(real(psik).*real(E[kdx])-imag(psik).*imag(E[kdx]))
    # end
    vk=push_particle5(xk,vk,E,(2*pi)/L,dt*rksc[rkdx])

    @acc xk+=(dt*rksd[rkdx]).*vk

  end
end
# MPI.Barrier(comm)

# if (mpi_rank==0)
toc()
print(fieldenergy)

# figure()
# ttime=(0:Nt-1)*dt
# semilogy(ttime,fieldenergy)
# grid()
# xlabel("time")
# ylabel("electrostatic energy")

# end


# MPI.Finalize()
