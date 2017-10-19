
using ArrayFire
a=rand(AFArray{Float64},2)
b=rand(AFArray{Float64},2)
c=zeros(AFArray{Complex128},2)
#This works
c=a+im*b
for idx=1:2
 c[idx]=a[idx]*2.0*im
end
#This doesnt
for idx=1:2
 c[idx]=a[idx]+im*b[idx]
end

using PyPlot
using ArrayFire
Np=Int(1e5)
Nf=Int(64)
k0=0.5
dt=0.1
tmax=15
L=2*pi/k0

#Set cpu
ArrayFire.setDevice(1)

function f(x,v)
  return exp(-0.5.*v.^2)/sqrt(2*pi).*(1+0.1.*cos(k0*x))
end

xk=rand(AFArray{Float64},Np,1)*L
vk=randn(AFArray{Float64},Np,1)
wk=f(xk,vk)*L


# Coefficients for Runge Kutta
rksd=[2/3.0, -2/3.0, 1  ]
rksc=[ 7/24.0, 3/4.0, -1/24.0]

kx=(1:Nf).''*2*pi/L

Nt=Int(ceil(tmax/dt))
fieldenergy=zeros(Float64,Nt,1)

tic()
for tdx=1:Nt

  @inbounds for rkdx=1:length(rksd)

    rhor=zeros(AFArray{Float64},Nf)
    rhoi=zeros(AFArray{Float64},Nf)
    @inbounds for kdx=1:Nf
      rhor[kdx]= sum( cos(2*pi/L*kdx*xk).*wk)
      rhoi[kdx]= sum( sin(2*pi/L*kdx*xk).*wk)

    end
    rho=rhor-im*rhoi


    E=-rho./(-im*kx)
    if rkdx==1
      fieldenergy[tdx]=(real(E'*E)*L*0.5*2)[1]
    end

    @inbounds for kdx=1:Nf
      psik=exp(im*2*pi/L*kdx*xk)
      vk+=dt*rksd[rkdx]*2*(real(psik).*real(E[kdx])-imag(psik).*imag(E[kdx]))
    end

    xk+=dt*rksc[rkdx]*vk

  end
end
toc()
ttime=(0:Nt-1)*dt


# figure()
# semilogy(ttime,fieldenergy)
# grid()
# xlabel("time")
# ylabel("electrostatic energy")
