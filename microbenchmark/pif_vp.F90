program simple_vp_pif
IMPLICIT NONE

REAL(8), PARAMETER :: PI =  3.1415926535897
integer(8), PARAMETER :: Np =1e5, Nf=64
real(8), parameter :: dt=0.1, k0=0.5, tmax=15
real(8) :: kx0,L
! Coefficients for Third order Runge Kutta
real(8), parameter, dimension(3) :: rksd= (/ 2.0/3.0, -2.0/3.0, 1.0  /)
real(8), parameter, dimension(3) :: rksc= (/ 7.0/24.0, 3.0/4.0, -1.0/24.0 /)
REAL(8), DIMENSION(:), ALLOCATABLE :: fieldenergy
integer(8) :: tdx, kdx, idx, rkdx, Nt
! Particles
real(8), dimension(Np) :: xn,vn,wn
integer(8) :: start, stop, rate
!
complex(8), dimension(Nf) :: rho, E
complex(8), dimension(Np) :: psin

! Set values
L=2.0*PI/k0
kx0=k0

Nt=floor(tmax/dt)
allocate(fieldenergy(1:Nt))

 
              
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
        do kdx=1,Nf
            rho(kdx)=sum( exp( DCMPLX(0.0, -xn*kdx*kx0  ) )*wn )/L/(1.0*Np)
        end do
        
        do kdx=1,Nf
            E(kdx) = -rho(kdx)/DCMPLX(0.0,- kx0*kdx)
        end do
        
        
        if (rkdx==1) then
            fieldenergy(tdx)=real(dot_product(E,E))/2.0
        end if

        do kdx=1,Nf
             psin=exp( DCMPLX(0.0, xn*kdx*kx0  ) )
            vn=vn+dt*rksc(rkdx)*2.0*(real(psin)*real(E(kdx)) - imag(psin)*imag(E(kdx)))
        end do

        xn=xn+dt*rksd(rkdx)*vn
        
        !print *, rksd(rkdx)
    end do


end do

call system_clock(stop,rate)

print *, real(stop-start)/real(rate)
! print '("Time = ",f8.4," seconds.")',finish-start

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





