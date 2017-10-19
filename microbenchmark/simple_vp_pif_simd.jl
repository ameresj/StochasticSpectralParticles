
Np=Int(1e5)
Nf=Int(64)
k0=0.5
dt=0.1
tmax=5
L=2*pi/k0


function f(x,v)
  return exp(-0.5.*v.^2)/sqrt(2*pi).*(1+0.1.*cos(k0*x))
end

xk=rand(Np,1)*L
vk=randn(Np,1)
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

    rho=Array(Complex128,Nf)

    rho = @parallel (+) for pdx=1:Np
    rhon=Array{Complex128}(Nf)
    @simd  for kdx=1:Nf
         @inbounds rhon[kdx]=exp(-im*kx[kdx]*xk[pdx])*wk[pdx]
      end
    rhon
    end

  rho=rho/L/Np

    E=-rho./(-im*kx)
    if rkdx==1
      fieldenergy[tdx]=(real(E'*E)*L*0.5*2)[1]
    end

    En=zeros(Float64,Np,1)
    @parallel for pdx=1:Np
      @simd for kdx=1:Nf
        @inbounds psik=exp(im*2*pi/L*kdx*xk[pdx])
        @inbounds En[pdx]+=(real(psik).*real(E[kdx])-imag(psik).*imag(E[kdx]))
      end
    end
    vk+=dt*rksd[rkdx]*2*En
    xk+=dt*rksc[rkdx]*vk

  end
end
toc()
# ttime=(0:Nt-1)*dt


# figure()
# semilogy(ttime,fieldenergy)
# grid()
# xlabel("time")
# ylabel("electrostatic energy")
