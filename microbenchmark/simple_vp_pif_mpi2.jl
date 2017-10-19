using PyPlot
Pkg.add("Yeppp")
using Devectorize
import MPI


using Yeppp

# x = rand(Float64,10^6)
# y=similar(x)
# ty = @elapsed Yeppp.log!(y, x)
# t  = @elapsed log(x)
#
# ty = @elapsed Yeppp.sin!(y, x)
# t  = @elapsed y=sin(x)
#
# t/ty

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

xk=rand(Np_loc,1)*L
vk=randn(Np_loc,1)
wk=f(xk,vk)*L


function  rhs_particle5(xk::Array{Float64,2},
         wk::Array{Float64,2}, Nf::Int, kx0::Float64)
    Np=length(xk)
    rho=zeros(Complex128,Nf)
    # sinn=1.0
    # cosn=1.0
    @inbounds for pdx=1:Np
      @inbounds arg=kx0*xk[pdx]


      psin1=(Yeppp.cos!(arg)- im*Yeppp.sin!(arg))
      psin=1.0+im*0.0
      for kdx=1:Nf
         psin=psin*psin1
         @inbounds rho[kdx]+=psin*wk[pdx]
       end
    end
  return rho
end
  rho=rhs_particle5(xk,wk,Nf,(2*pi)/L)
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


# Coefficients for Runge Kutta
const rksd=[2/3.0, -2/3.0, 1  ]
const rksc=[ 7/24.0, 3/4.0, -1/24.0]

kx=(1:Nf).''*2*pi/L

Nt=Int(ceil(tmax/dt))
fieldenergy=zeros(Float64,Nt,1)
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
     vk=push_particle5(xk,vk,E,(2*pi)/L,dt*rksd[rkdx])

    xk+=(dt*rksc[rkdx])*vk

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
