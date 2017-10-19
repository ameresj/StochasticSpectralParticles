# using Devectorize
import MPI

using PyPlot



Np=Int(1e5)
Nf=Int(64) # Number of Fourier modes
k0=0.5
dt=0.1
tmax=15

L=2*pi/k0


function f(x,v)
  return exp(-0.5.*v.^2)/sqrt(2*pi).*(1+0.1.*cos(k0*x))
end

MPI.Init()

comm = MPI.COMM_WORLD
MPI.Barrier(comm)
mpi_rank = MPI.Comm_rank(comm)
mpi_size = MPI.Comm_size(comm)
print(mpi_size)

Np_loc=Int(floor(Np/mpi_size))

#Seed the random generator
#We would like to skip ahead values
srand( (Np_loc*4)*(mpi_rank+1) )

xk=rand(Np_loc)*L
vk=randn(Np_loc)
wk=f(xk,vk)*L

@inbounds function  rhs_particle{T}(xk::Array{T,1},vk::Array{T,1},
                wk::Array{T,1},Nf::Int,kx0::T)
  rho=Array(Complex{T},Nf)
  for kdx=1:Nf
    rho[kdx]=sum(exp(-im*kx0*kdx*xk).*wk)
  end
  return rho
end

function  rhs_particle2{T}(xk::Array{T,1}, vk::Array{T,1},
                wk::Array{T,1}, Nf::Int, kx0::T)
    rho=zeros(Complex{T},Nf)
    @inbounds for pdx=1:length(xk)
    @simd for kdx=1:Nf
         rho[kdx]+=exp(-im*kx0*xk[pdx]*kdx)*wk[pdx]
    end
    end
  return rho
end
function  rhs_particle4{T}(xk::Array{T,1}, vk::Array{T,1},
                wk::Array{T,1}, Nf::Int, kx0::T)
    rho=zeros(Complex{T},Nf)
    @inbounds for pdx=1:length(xk)
    @simd for kdx=1:Nf
        arg::T=kx0*xk[pdx]*kdx
        rho[kdx]+=(cos(arg)-im*sin(arg))*wk[pdx]
    end
    end
  return rho
end
@inbounds function  rhs_particle5{T}(xk::Array{T,1},
         wk::Array{T,1}, Nf::Int, kx0::T)
    rho=zeros(Complex{T},Nf)
    for pdx=1:length(xk)
      arg::T=kx0*xk[pdx]
      psin1::Complex{T}=(cos(arg)- im*sin(arg))
      psin::Complex{T}=wk[pdx]
      for kdx=1:Nf
         psin*=psin1
         rho[kdx]+=psin
       end
    end
  return rho
end

@inbounds function  rhs_particle6{T}(xk::Array{T,1},
         wk::Array{T,1}, Nf::Int, kx0::T)
    rho=zeros(Complex{T},Nf)
    for pdx=1:length(xk)
      arg::T=kx0*xk[pdx]
      psin1::Complex{T}=(cos(arg)- im*sin(arg))
      psin2::Complex{T}=psin1^2
      psin3::Complex{T}=psin2*psin1
      psin4::Complex{T}=psin2^2
      psin_::Complex{T}=psin4
      psin1*=wk[pdx]
      psin2*=wk[pdx]
      psin3*=wk[pdx]
      psin4*=wk[pdx]
      for kdx=0:Int(Nf/4)-1
         rho[kdx*4+1]+=psin1
         rho[kdx*4+2]+=psin2
         rho[kdx*4+3]+=psin3
         rho[kdx*4+4]+=psin4
         psin1*=psin_
         psin2*=psin_
         psin3*=psin_
         psin4*=psin_
       end
    end
  return rho
end
@inbounds function  rhs_particle7{T}(xk::Array{T,1},
         wk::Array{T,1}, Nf::Int, kx0::T)
    rho=zeros(Complex{T},Nf)
    for pdx=1:length(xk)
      arg::T=kx0*xk[pdx]
      psin1::Complex{T}=(cos(arg)- im*sin(arg))
      psin2::Complex{T}=psin1^2
      psin4::Complex{T}=psin2^2
      psin8::Complex{T}=psin4^2
      psin_::Complex{T}=psin4
      psin1*=wk[pdx]
      psin3::Complex{T}=psin1*psin2
      psin5::Complex{T}=psin4*psin1
      psin7::Complex{T}=psin4*psin3
      psin2*=wk[pdx]
      psin6::Complex{T}=psin2*psin4
      psin4*=wk[pdx]
      psin8*=wk[pdx]

      for kdx=0:Int(Nf/8)-1
         idx::Int=kdx*8
         rho[idx+1]+=psin1
         rho[idx+2]+=psin2
         rho[idx+3]+=psin3
         rho[idx+4]+=psin4
         rho[idx+5]+=psin1
         rho[idx+6]+=psin2
         rho[idx+7]+=psin3
         rho[idx+8]+=psin4
         psin1*=psin_
         psin2*=psin_
         psin3*=psin_
         psin4*=psin_
         psin5*=psin_
         psin6*=psin_
         psin7*=psin_
         psin8*=psin_
       end
    end
  return rho
end
function  push_particle{T}(xk::Array{T,1}, vk::Array{T,1},
                  E::Array{Complex{T},1}, kx0::T, dt::T)

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

@inbounds function  push_particle2{T}(xk::Array{T,1}, vk::Array{T,1},
                  E::Array{Complex{T},1}, kx0::T, dt::T)
    for pdx=1:length(xk)
        @simd for kdx=1:length(E)
        arg::T=kx0*xk[pdx]*kdx
        vk[pdx]+=dt*2*(cos(arg)*real(E[kdx])-sin(arg)*imag(E[kdx]))
      end
    end
    nothing
end

@inbounds function  push_particle5{T}(xk::Array{T,1}, vk::Array{T,1},
                  E::Array{Complex{T},1}, kx0::T, dt::T)

    for pdx=1:length(xk)
      arg::T=kx0*xk[pdx]
      psin1::Complex{T}=(cos(arg)+ im*sin(arg))
      psin::Complex{T}=1.0
      for kdx=1:length(E)
          psin=psin*psin1
         @inbounds vk[pdx]+=dt*2*(real(psin)*real(E[kdx])-imag(psin)*imag(E[kdx]))
      end
    end
    nothing
end


function  rhs_particle3{T}(xk::Array{T,1},vk::Array{T,1},
                wk::Array{T,1},Nf::Int,kx0::T)

  rho=Array(Complex{T},Nf)
  rho = @parallel (+) for pdx=1:length(xk)
    rhon=Array(Complex{T},Nf)
    @simd  for kdx=1:Nf
         @inbounds rhon[kdx]=exp(-im*kx0*kdx*xk[pdx])*wk[pdx]
      end
    rhon
  end
  return rho
end
# using BenchmarkTools
# Nf=256
# @benchmark rho=rhs_particle5(xk,wk,Nf,(2*pi)/L)
#
# @benchmark rho6=rhs_particle6(xk,wk,Nf,(2*pi)/L)
#
# @benchmark rho6=rhs_particle7(xk,wk,Nf,(2*pi)/L)
#
#  rho6=rhs_particle6(xk,wk,Nf,(2*pi)/L)
#   rho=rhs_particle5(xk,wk,Nf,(2*pi)/L)
# maximum(abs(rho6-rho))

# code_llvm(rhs_particle4, (Array{T,1},
#    Array{T,1},Array{T,1},Int,Float64) )

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

kx=(1:Nf)*2*pi/L

Nt=Int(ceil(tmax/dt))
fieldenergy=zeros(Float64,Nt)
MPI.Barrier(comm)
if (mpi_rank==0)
tic()
end
for tdx=1:Nt

  @inbounds for rkdx=1:length(rksd)

    rho=rhs_particle5(xk,wk,Nf,(2*pi)/L)
    # rho=zeros(Complex128,Nf)
    # @inbounds for kdx=1:Nf
    #   rho[kdx]=sum(exp(-im*2*pi/L*kdx*xk).*wk)
    # end

    #Allreduce over a vector
    rho=MPI.allreduce(rho, MPI.SUM, comm)

    rho=rho/L/Np

    E=-rho./(-im*kx)
    if rkdx==1
      fieldenergy[tdx]=(real(E'*E)*L*0.5*2)[1]
    end

    # @inbounds for kdx=1:Nf
    #   psik=exp(im*2*pi/L*kdx*xk)
    #   vk+=dt*rksd[rkdx]*2*(real(psik).*real(E[kdx])-imag(psik).*imag(E[kdx]))
    # end
     push_particle5(xk,vk,E,(2*pi)/L,dt*rksc[rkdx])

    xk.+=(dt*rksd[rkdx])*vk

  end
end
MPI.Barrier(comm)

if (mpi_rank==0)
   toc()
  print(fieldenergy)

figure()
ttime=(0:Nt-1)*dt
semilogy(ttime,fieldenergy)
grid()
xlabel("time")
ylabel("electrostatic energy")

end


MPI.Finalize()
