program simple_vp_pif
IMPLICIT NONE

REAL(8), PARAMETER :: PI =  3.1415926535897
integer(8), PARAMETER :: Np =1e5, Nf=64
real(8), parameter :: dt=0.1, k0=0.1, tmax=15
real(8) :: kx0,L
! Coefficients for Third order Runge Kutta
real(8), parameter, dimension(3) :: rksd= (/ 2.0/3.0, -2.0/3.0, 1.0  /)
real(8), parameter, dimension(3) :: rksc= (/ 7.0/24.0, 3.0/4.0, -1.0/24.0 /)
REAL(8), DIMENSION(:), allocatable :: fieldenergy
integer(8) :: tdx, kdx, idx, rkdx, Nt,pdx
! Particles
double precision, dimension(:), allocatable :: xn,vn,wn
integer(8) :: start, stop, rate
real(8) ::Exn, xx, ww,arg
real(8) :: delta
!
complex(8), dimension(Nf) :: rho, E, rhon
complex(8) :: psin
real(8), dimension(Nf) :: kx

! Set values
L=2.0*PI/k0
kx0=k0

do kdx=1,Nf
    kx(kdx)=k0*kdx
end do

Nt=floor(tmax/dt)
allocate(fieldenergy(1:Nt))
 
 allocate(xn(1:Np))
 allocate(vn(1:Np))
 allocate(wn(1:Np))
              
! Sample particles
call random_number(xn)
call random_number(vn) ! Should be normally distributed
xn=xn*L
vn=vn*10.0 - 5.0
wn=f0(xn,vn,k0)*L*10.0

call system_clock(start,rate)
                  
do tdx = 1, Nt
    do rkdx = 1, size(rksd)
        
        ! Accumulate charge
        rho=0.0
            !$OMP PARALLEL DO SIMD reduction(+:rho) private(rhon,xx,ww,kdx)
            do pdx=1,Np
                    ww=wn(pdx)
                    xx=xn(pdx)
                    !$OMP SIMD aligned(vn:64) aligned(xn:64)
                    do kdx=1,Nf 
                        rhon(kdx)=exp( DCMPLX(0.0, -xx*kx0*kdx ) )*ww
                        !arg=xx*kx0*kdx
                        !rhon(kdx)=DCMPLX(cos( arg), -sin( arg))*ww
                    end do
                    rho=rho+rhon
            end do
            !$OMP END PARALLEL DO SIMD
 
!         rho=rho/L/(1.0*real(Np))
        
        !$OMP SIMD
        do kdx=1,Nf
            E(kdx) = -rho(kdx)/DCMPLX(0.0,- kx0*kdx)
        end do
        
        
!         if (rkdx==1) then
!             fieldenergy(tdx)=real(dot_product(E,E))/2.0
!         end if
!         
!         !$OMP PARALLEL DO private(xx,Exn,psin)
!         do pdx=1,Np 
!             xx=xn(pdx)
!             Exn=0.0
!             !$OMP SIMD reduction(+:Exn) private(psin)
!             do kdx=1,Nf   
!                  psin=exp( DCMPLX(0.0, xx*kx0*kdx  ) )
!                  Exn=Exn+2.0*(real(psin)*real(E(kdx)) - imag(psin)*imag(E(kdx)))
!                   !arg= (xx*kx0*kdx  )
!                  !Exn=Exn+2.0*(cos(arg)*real(E(kdx)) - sin(arg)*imag(E(kdx)))
!             end do
!             
!             vn(pdx)=vn(pdx)+dt*rksc(rkdx)*Exn
!             
!         end do
!         !$OMP END PARALLEL DO
!         
        delta=dt*rksd(rkdx)
        !$OMP PARALLEL DO SIMD aligned(vn:64) aligned(xn:64)
        do pdx=1,Np       
            xn(pdx)=xn(pdx)+delta*vn(pdx)
        end do
        !$OMP END PARALLEL DO SIMD
        
        !!!print *, rksd(rkdx)
        end do


end do

call system_clock(stop,rate)

print *, real(stop-start)/real(rate)
! print *, fieldenergy

contains



function f0(x,v,k0)  result(fun)
  real(8), dimension(:), intent(in) :: x,v
  real(8), intent(in) :: k0
  real(8), dimension(size(x)) :: fun
  fun=exp(-0.5*v**2)/sqrt(2*PI)*(1.0+0.1*cos(k0*x))
  
end function f0

! subroutine rgauss(sigma, y1,y2)
! real(8):: x1, x2, w, 
! real(8), intent(in) :: y1, y2
! real(8)::sigma
! 
! do while ( (w .ge. 1.0).or.(w.eq.0) )
! x1 = 2.0 * rand(0) - 1.0
! x2 = 2.0 * rand(0) - 1.0
! w = x1 * x1 + x2 * x2
! end do
! 
! w = sigma*sqrt( (-2.0 * log( w ) ) / w )
! y1 = x1 * w
! y2 = x2 * w
! 
! end subroutine rgauss

end program





