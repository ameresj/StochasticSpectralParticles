program pif_vp_aosa2
IMPLICIT NONE

! Test Arrays of structures of arrays
#define CHUNKSIZE 2
TYPE chunk_particle
  REAL(8), dimension(CHUNKSIZE)  :: x, v,w
ENDTYPE chunk_particle


REAL(8), PARAMETER :: PI =  3.1415926535897
integer(8), PARAMETER :: Np =1e4, Nf=64
real(8), parameter :: dt=0.1, k0=0.1, tmax=15
real(8) :: kx0,L
! Coefficients for Third order Runge Kutta
real(8), parameter, dimension(3) :: rksd= (/ 2.0/3.0, -2.0/3.0, 1.0  /)
real(8), parameter, dimension(3) :: rksc= (/ 7.0/24.0, 3.0/4.0, -1.0/24.0 /)
REAL(8), DIMENSION(:), allocatable :: fieldenergy
integer(8) :: tdx, kdx, idx, rkdx, Nt,pdx, Nch,cdx
! Particles
real(8), dimension(CHUNKSIZE) :: xn,vn,wn
integer(8) :: start, stop, rate
real(8) ::Exn, xx, ww,arg, Ekr, Eki
real(8), dimension(CHUNKSIZE) :: Ex
!
complex(8), dimension(Nf) :: rho, E, rhon
complex(8) :: psin, rhonk
real(8), dimension(Nf) :: kx

type(chunk_particle), dimension(:),allocatable :: particles


! Set values
L=2.0*PI/k0
kx0=k0

do kdx=1,Nf
    kx(kdx)=k0*kdx
end do

Nt=floor(tmax/dt)
allocate(fieldenergy(1:Nt))
 
! Number of CHUNKSIZE
Nch=floor(Np/(CHUNKSIZE*1.0))
allocate(particles(1:Nch))

do cdx=1,Nch
    call random_number(particles(cdx)%x)
    call random_number(particles(cdx)%v)
    particles(cdx)%x=particles(cdx)%x*L
    particles(cdx)%v=particles(cdx)%v*10.0 - 5.0
    particles(cdx)%w=f0(particles(cdx)%x,particles(cdx)%v,k0)*L*10.0
end do


call system_clock(start,rate)
                  
do tdx = 1, Nt
    do rkdx = 1, size(rksd)
        
        ! Accumulate charge
        rho=0.0
             !$OMP PARALLEL DO reduction(+:rho) private(rhon)
            do cdx=1,Nch
                !unpack particle
                call accum_mode(rhon, particles(cdx), kx0,Nf)
                rho=rho+rhon
            end do
            !$OMP END PARALLEL DO

        rho=rho/L/(1.0*real(Np))
        
        !$OMP SIMD
        do kdx=1,Nf
            E(kdx) = -rho(kdx)/dcmplx(0.0,- kx0*kdx)
        end do
        
        
        if (rkdx==1) then
            fieldenergy(tdx)=real(dot_product(E,E))/2.0
        end if
        
!         !$OMP PARALLEL DO private(xx,Exn,psin,xn,vn) 
!         do cdx=1,Nch
!                !unpack particle
!                 xn=particles(cdx)%x
!                 vn=particles(cdx)%v
!             
!             Ex=0
!             !$OMP SIMD reduction(+:Ex) private(psin,Ekr,Eki)    
!             do kdx=1,Nf                
!                 Ekr=real(E(kdx))
!                 Eki=imag(E(kdx))
!                 do pdx=1,CHUNKSIZE
!                     xx=xn(pdx)
!                     psin=exp( dcmplx(0.0, xx*kx0*kdx  ) )
!                     Ex(pdx)+=2.0*(real(psin)*Ekr - imag(psin)*Eki)
!                 end do
!             end do
!             
!             do pdx=1,CHUNKSIZE
!                 vn(pdx)=vn(pdx)+dt*rksc(rkdx)*Ex(pdx)
!                 xn(pdx)=xn(pdx)+dt*rksd(rkdx)*vn(pdx)
!             end do
!             
!             particles(cdx)%v=vn
!             particles(cdx)%x=xn
! 
!         end do
!         !$OMP END PARALLEL DO
!         
        
        
!         !$OMP PARALLEL DO private(xn,vn)
!         do cdx=1,Nch
!                !unpack particle
!                 xn=particles(cdx)%x
!                 vn=particles(cdx)%v
!                 !$OMP SIMD
!                 do pdx=1,CHUNKSIZE        
!                     xn(pdx)=xn(pdx)+dt*rksd(rkdx)*vn(pdx)
!                 end do
!                 particles(cdx)%x=xn
!         end do
!         !$OMP END PARALLEL DO
        
        !$OMP PARALLEL DO private(xn,vn)
        do cdx=1,Nch
            call push_x(particles(cdx),  dt*rksd(rkdx))
        end do
        !$OMP END PARALLEL DO
        
        !!!print *, rksd(rkdx)
        end do


end do

call system_clock(stop,rate)

print *, real(stop-start)/real(rate)
! print *, fieldenergy

contains


subroutine push_x(chunk, dt)
type(chunk_particle), intent(inout):: chunk
real(8),intent(in) :: dt
integer :: pdx
! real(8), dimension(CHUNKSIZE) :: xn, vn

!associate( xn => chunk%x, vn=>chunk%v)
do pdx=1,CHUNKSIZE
        !xn(pdx)=xn(pdx)+dt*vn(pdx)
        chunk%x(pdx)=chunk%x(pdx)+dt*chunk%v(pdx)
end do
!end associate

! xn=chunk%x
! vn=chunk%v
! !$OMP SIMD
! do pdx=1,CHUNKSIZE
!         xn(pdx)=xn(pdx)+dt*vn(pdx)
! end do
! chunk%x=xn
end subroutine


subroutine accum_mode(rhs, chunk, kx0,Nf)
integer(8), intent(in) :: Nf
real(8), intent(in) ::kx0
type(chunk_particle), intent(in):: chunk
! real(8), dimension(CHUNKSIZE) :: xn, vn, wn
complex(8),dimension(Nf), intent(out) :: rhs
complex(8) :: rhonk
real(8) :: arg
integer :: kdx,pdx
real(8), dimension(CHUNKSIZE) :: xn, vn

xn=chunk%x
! vn=chunk%v
wn=chunk%w

!!!$OMP SIMD reduction(+:rhonk)
!$OMP SIMD private(arg,rhonk)
do kdx=1,Nf 
    rhonk=dcmplx(0.0,0.0)
    do pdx=1,CHUNKSIZE
            arg=xn(pdx)*kx0*kdx
            rhonk=rhonk+exp(dcmplx(0.0, -arg ))*wn(pdx)
            !rhonk=rhonk+dcmplx(cos(arg), -sin(arg) )*chunk%w(pdx)
    end do
    rhs(kdx)=rhonk
end do

end subroutine


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





