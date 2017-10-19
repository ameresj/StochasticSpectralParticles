using Yeppp
import MPI

Np=Int(1e5)
Nf=Int(64)
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

#using BenchmarkTools


# sinsum=zeros(Float64,Np)
# s=zeros(Float64,1)
kx0=2*pi/L

@inbounds function  rhs_particle{T}(xk::Array{T,1},
         wk::Array{T,1}, Nf::Int, kx0::T)

    rhoc=zeros(T,Nf)
    rhos=zeros(T,Nf)
    rho=zeros(Complex{T},Nf)
    for kdx=1:Nf
        k::T=kx0*kdx
        rhoc[kdx]=Yeppp.dot(Yeppp.cos(xk.*k),wk)
        rhos[kdx]=Yeppp.dot(Yeppp.sin(xk.*k),wk)
    end

    rho.=rhoc-im.*rhos
  return rho
end

# @inbounds function  rhs_particle2{T}(xk::Array{T,1},
#          wk::Array{T,1}, Nf::Int, kx0::T)
#
#     rhoc=zeros(T,Nf)
#     rhos=zeros(T,Nf)
#     rho=zeros(Complex{T},Nf)
#     psic0=Yeppp.cos(xk*kx0)
#     psis0=Yeppp.sin(xk*kx0)
#     psicn=ones(T,length(xk))
#     psisn=zeros(T,length(xk))
#     tmp=zeros(T,length(xk))
#
#     for kdx=1:Nf
#         #psicn.= psicn.*psic0 .- psisn.*psis0
#         #psisn.= psicn.*psis0 .+ psisn.*psic0
#         tmp=Yeppp.subtract(Yeppp.multiply(psicn,psic0),Yeppp.multiply(psisn,psis0))
#         psisn=Yeppp.add!(psisn,Yeppp.multiply(psicn,psis0),Yeppp.multiply(psisn,psic0))
#         psicn=tmp
#         rhoc[kdx]=Yeppp.dot(psicn,wk)
#         rhos[kdx]=Yeppp.dot(psisn,wk)
#     end
#
#     rho.=rhoc-im.*rhos
#   return rho
# end
#
function  rhs_particle5{T}(xk::Array{T,1},
         wk::Array{T,1}, Nf::Int, kx0::T)
    Np=length(xk)
    rho=zeros(Complex{T},Nf)
    @inbounds for pdx=1:Np
      @inbounds arg=kx0*xk[pdx]
      psin1::Complex{T}=(cos(arg)- im*sin(arg))
      psin::Complex{T}=wk[pdx]
      for kdx=1:Nf
         psin=psin*psin1
         @inbounds rho[kdx]+=psin
       end
    end
  return rho
end
#
# function  rhs_particle2{T}(xk::Array{T,1},
#          wk::Array{T,1}, Nf::Int, kx0::T)
#     rho=zeros(Complex{T},Nf)
#     const N::Int=10000
#     Np=length(xk)
#     psin=ones(Complex{T},N)
#     psin1=Array(Complex{T},N)
#
#
#     for idx=1:round(Np/N)
#       nmin::Int=1+(idx-1)*N
#       nmax::Int=idx*N
#       psin.=wk[nmin:nmax]
#       #psin1=exp.(-im*kx0*xk)
#       psin1=(Yeppp.cos(kx0*xk[nmin:nmax])- im*Yeppp.sin(kx0*xk[nmin:nmax]))
#       for kdx=1:Nf
#          psin.*=psin1
#          @inbounds rho[kdx]+=sum(psin)
#        end
#     end
#   return rho
# end


# function  rhs_particle6{T}(xk::Array{T,1},
#          wk::Array{T,1}, Nf::Int, kx0::T)
#     Np=length(xk)
#     rho=zeros(Complex{T},Nf)
#     @inbounds for pdx=1:Np
#       @inbounds arg=kx0*xk[pdx]
#       psin1=(Yeppp.cos(arg)- im*Yeppp.sin(arg))
#       psin=1.0+im*0.0
#       for kdx=1:Nf
#          psin=psin*psin1
#          @inbounds rho[kdx]+=psin*wk[pdx]
#        end
#     end
#   return rho
# end
@inbounds function push_particle{T}(xk::Array{T,1},
              vk::Array{T,1}, E::Array{Complex{T},1},
                kx0::T, tauv::T,taux::T )

    Er=real(E)*2*tauv; Ei=-imag(E)*2*tauv;
    for kdx=1:Nf
        k::T=kx0*kdx
        Yeppp.add!(vk,vk, Er[kdx]*Yeppp.cos(xk*kx0)  )
        Yeppp.add!(vk,vk, Ei[kdx]*Yeppp.sin(xk*kx0)  )
    end

    Yeppp.add!(xk,xk,vk.*taux)
    return xk, vk
end
# using BenchmarkTools
# @benchmark rho=rhs_particle(xk,wk,Nf,(2*pi)/L)/Np/L
#
# @benchmark rho2=rhs_particle2(xk,wk,Nf,(2*pi)/L)/Np/L
#
# @benchmark rho5=rhs_particle5(xk,wk,Nf,(2*pi)/L)/Np/L
# @benchmark rho6=rhs_particle6(xk,wk,Nf,(2*pi)/L)/Np/L
# Coefficients for Runge Kutta
rksd=[2/3.0, -2/3.0, 1  ]
rksc=[ 7/24.0, 3/4.0, -1/24.0]

kx=collect((1:Nf)*2*pi/L)

Nt=Int(ceil(tmax/dt))
fieldenergy=zeros(Float64,Nt)
kineticenergy=zeros(Float64,Nt)
E=zeros(Complex{Float64},Nf)

MPI.Barrier(comm)
if (mpi_rank==0)
tic()
end
for tdx=1:Nt

  @inbounds for rkdx=1:length(rksd)


    rho=rhs_particle(xk,wk,Nf,(2*pi)/L)/Np/L
    #Allreduce over a vector
    rho=MPI.allreduce(rho, MPI.SUM, comm)

    E=-rho./(-im*kx)
    if rkdx==1
      fieldenergy[tdx]=(real(E'*E)*L*0.5*2)[1]
      kineticenergy[tdx]=0.5*Yeppp.dot(vk.^2,wk)/Np
    end

    push_particle(xk,vk, E,(2*pi)/L, rksc[rkdx]*dt,rksd[rkdx]*dt)

  end
end
ttime=(0:Nt-1)*dt
if (mpi_rank==0)
   toc()

energy=fieldenergy+kineticenergy
print(norm(energy-energy[1]))

using PyPlot
figure()
semilogy(ttime,fieldenergy)
grid()
xlabel("time")
ylabel("electrostatic energy")



figure()
semilogy(ttime,abs((energy-energy[1])/energy[1]))
grid()
xlabel("time")
ylabel("energy error")
end
MPI.Finalize()
