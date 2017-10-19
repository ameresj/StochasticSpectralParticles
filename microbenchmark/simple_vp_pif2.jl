
using PyPlot
Np=Int(1e5)
Nf=Int(2)
k0=0.5
dt=0.1
tmax=15
L=2*pi/k0


function f(x,v)
  return exp(-0.5.*v.^2)/sqrt(2*pi).*(1+0.1.*cos(k0*x))
end

xk=rand(Np)*L
vk=randn(Np)
wk=f(xk,vk)*L


# Coefficients for Runge Kutta
rksd=[2/3.0, -2/3.0, 1  ]
rksc=[ 7/24.0, 3/4.0, -1/24.0]

kx=collect(1:Nf)*2*pi/L

Nt=Int(ceil(tmax/dt))
fieldenergy=zeros(Float64,Nt)
kdx=1


tic()
@inbounds begin
for tdx=1:Nt

  for rkdx=1:length(rksd)

    rho=zeros(Complex128,Nf)
    for kdx=1:Nf
      rho[kdx]=sum(exp(-im*2*pi/L*kdx*xk).*wk)/L/Np
      #rho[kdx]=mapreduce(arg->begin x,v=arg; exp(-im*2*pi/L*kdx*x)*v  end,
      #             +, 0.0, zip(xk, vk))/L/Np
      #mapreduce(arg->exp(-im*2*pi/L*kdx*arg[1]).*arg[2],+,0.0, [xk,wk])/L/Np
    end
    E=-rho./(-im*kx)



    if rkdx==1
      fieldenergy[tdx]=(real(E'*E)*L*0.5*2)[1]
    end

    for kdx=1:Nf
      psik=exp(im*2*pi/L*kdx*xk)
      vk+=dt*rksc[rkdx]*2*(real(psik).*real(E[kdx])-imag(psik).*imag(E[kdx]))
      #vk=map( (v,x)-> v+dt*rksc[rkdx]*2*
      #       (cos(2*pi/L*kdx*x)*real(E[kdx])-sin(2*pi/L*kdx*x)*imag(E[kdx])),
      #    vk,xk)
    end

    xk.+=dt*rksd[rkdx]*vk

  end
end
end
toc()
ttime=(0:Nt-1)*dt

# energy=fieldenergy+kineticenergy
# print(norm(energy-energy[1]))

figure()
semilogy(ttime,fieldenergy)
grid()
xlabel("time")
ylabel("electrostatic energy")
