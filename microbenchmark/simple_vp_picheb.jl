using FastTransforms: cheb2leg, leg2cheb
using ApproxFun
#using Plots
Np=Int(1e5)
Nx=Int(ceil(64*pi/2))
k0=0.5
dt=0.1
tmax=15
L=2*pi/k0


function f(x,v)
  return exp(-0.5.*v.^2)/sqrt(2*pi).*(1+0.1.*cos(k0*x))
end

xk=rand(Np)*L
vk=randn(Np)
fk=f(xk,vk)
wk=f(xk,vk)*L

# Derivative of a chebyshev series
function chebTdx_series{T}(x::T, c::Array{T,1})
  N=length(c)-1;
  b1::T=0
  b2::T=0
  @inbounds for n=N:-1:2
     b0::T=c[n+1]+ 2*x*(n+1)/n*b1 - (n+2)/n*b2
    b2=b1; b1=b0;
  end
  return c[2]+ 4*x.*b1 - 3*b2
end

# Legendre polynomials
function LegendreP{T}(x::T,N::Int)
  Pn=Array(T, N)
  Pn[1]=1
  Pn[2]=x
  @inbounds for idx=3:N
    n=idx-2
    Pn[idx]= x.*(2*n+1)/(n+1).*Pn[idx-1]- n/(n+1).*Pn[idx-2]
  end
  return Pn
end


#@time LegendreP(xk[1],32)
function  rhs_particle_legendre{T}(xk::Array{T,1},wk::Array{T,1}, Nx::Int)
    Np=length(xk)
    rho=zeros(T,Nx)

    @inbounds for pdx=1:Np
      P0::T=1.0; P1::T=xk[pdx];
      rho[1]+=P0.*wk[pdx]
      rho[2]+=P1.*wk[pdx]
      @inbounds for idx=3:Nx
        n=idx-2
        P2::T=xk[idx].*(2*n+1)/(n+1).*P1- n/(n+1).*P0
        P0=P1; P1=P2;
        rho[idx]+=P2.*wk[pdx]
      end
    end
  return rho
end

function  rhs_particle2{T}(xk::Array{T,1},wk::Array{T,1}, Nx::Int)
    Np=length(xk)
    rho=zeros(T,Nx)
    Pn=Array(T, Nx)
    Pn[1]=1.0

    @inbounds for pdx=1:Np
      Pn[2]=xk[pdx]
      @inbounds for idx=3:Nx
        n=idx-2
        Pn[idx]= Pn[2].*(2*n+1)/(n+1).*Pn[idx-1]- n/(n+1).*Pn[idx-2]
      end
      rho+=Pn.*wk[pdx]
    end
  return rho
end

function  rhs_particle{T}(xk::Array{T,1},wk::Array{T,1}, Nx::Int)
    Np=length(xk)
    rho=zeros(T,Nx)
    @inbounds for pdx=1:Np
      rho+=LegendreP(xk[pdx],Nx).*wk[pdx]
    end
  return rho
end

#using BenchmarkTools
#t=@benchmark rhs=rhs_particle( xk/L*2-1,wk,Nx)/Np*(2/L)
#t2=@benchmark rhs=rhs_particle2( xk/L*2-1,wk,Nx)/Np*(2/L)
#t=@benchmark rhs2=rhs_particle_legendre( xk/L*2-1,wk,Nx)/Np*(2/L)




OM=Interval(0,L) # Domain
S=ApproxFun.Space(OM) #Default chebyshev space
B=periodic(S,0) # Periodic boundary
B2=DefiniteIntegral(S) # Nullspace condition
D2=-Derivative(S,2) #Laplace operator
D1=Derivative(S,1) # First derivative for H1 norm
IN=DefiniteIntegral(S) # First derivative for L2 norm

#Phi = \([B;B2;D2],[0.0;0.0;rho];tolerance=1E-16,maxlength=Nx+2)
QR = qrfact([B;B2;D2])

# Coefficients for Runge Kutta
rksd=[2/3.0, -2/3.0, 1  ]
rksc=[ 7/24.0, 3/4.0, -1/24.0]
#DefiniteIntegral(S)

Nt=Int(ceil(tmax/dt))
fieldenergy=zeros(Float64,Nt)
kineticenergy=zeros(Float64,Nt)
momentum=zeros(Float64,Nt)
rho=Fun(S)
Phi=Fun(S)
tic()
@inbounds for tdx=1:Nt

  @inbounds for rkdx=1:length(rksd)

    # Bin to right hand side
    rhs=rhs_particle( xk/L*2-1,wk,Nx)/Np*(2/L)
    #rhs=rhs_particle_legendre( xk/L*2-1,wk,Nx)/Np*(2/L)
    # Legendre L^2 projection
    rhs=rhs./(2./(2*(0:Nx-1)+1.0))
    # Remove ion background (quick'n dirty)
    rhs[1]=0
    #Transform to Chebyshev
    rho=Fun(S,leg2cheb(rhs))

    # Solve periodic Poisson equation with Chebyshev
    # Chebyshev periodic Poisson
    #Phi = \([B;B2;D2],[0.0;0.0;rho];tolerance=1E-16)
    Phi=QR \[0.0;0.0;rho]
    #,maxlength=Nx+2

    #E=-rho./(-im*kx)
    if rkdx==1
      fieldenergy[tdx]=coefficients(IN*(D1*Phi)^2)[1]/2
      #kineticenergy[tdx]=0.5.*sum(vk.^2.*wk )/Np
      #momentum[tdx]=sum(vk.*wk)/Np
    end
    Phic=-coefficients(Phi)[1:Nx+2]/L*2
    vk.+=dt*rksc[rkdx].*map(x->chebTdx_series(x, Phic), xk./(L*2.)-1.0)
    xk.+=dt*rksd[rkdx].*vk
    xk=mod(xk,L)
  end

  #print(tdx)
end
toc()
ttime=(0:Nt-1)*dt

#print(ttime)
#plot(Phi)
using PyPlot: semilogy,plot

semilogy(ttime,fieldenergy)
#semilogy(ttime,kineticenergy)
#semilogy(ttime,abs(momentum-momentum[1]))
#PyPlot.grid()
#norm(Phi)
#plot(ttime,momentum/Np)
#scatter(xk,vk,zcolor=fk)

# QR*Phi
#Phi





# figure()
# semilogy(ttime,fieldenergy)
# grid()
# xlabel("time")
# ylabel("electrostatic energy")
#using Plots
#plot(Phi)
