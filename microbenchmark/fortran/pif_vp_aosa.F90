program pif_vp_aosa
IMPLICIT NONE

! Test Arrays of structures of arrays
#define CHUNKSIZE 16
TYPE chunk_particle
  REAL(8), dimension(CHUNKSIZE)  :: x, v,w
ENDTYPE chunk_particle


REAL(8), PARAMETER :: PI =  3.1415926535897
integer(8), PARAMETER :: Np =1e5, Nf=64
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
real(8) ::Exn, xx, ww,arg
!
complex(8), dimension(Nf) :: rho, E, rhon
complex(8) :: psin
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
            !$OMP PARALLEL DO reduction(+:rho) private(rhon,xx,ww,kdx,wn,xn)
            do cdx=1,Nch
                !unpack particle
                xn=particles(cdx)%x
                vn=particles(cdx)%v
                wn=particles(cdx)%w
                do pdx=1,CHUNKSIZE
                        ww=wn(pdx)
                        xx=xn(pdx)
                        !$OMP SIMD
                        do kdx=1,Nf 
                            rhon(kdx)=exp( CMPLX(0.0, -xx*kx0*kdx ) )*ww
                        end do
                        rho=rho+rhon
                end do
            end do
            !$OMP END PARALLEL DO

        rho=rho/L/(1.0*real(Np))
        
        !$OMP SIMD
        do kdx=1,Nf
            E(kdx) = -rho(kdx)/CMPLX(0.0,- kx0*kdx)
        end do
        
        
        if (rkdx==1) then
            fieldenergy(tdx)=real(dot_product(E,E))/2.0
        end if
        
        !$OMP PARALLEL DO private(xx,Exn,psin,xn,vn) 
        do cdx=1,Nch
               !unpack particle
                xn=particles(cdx)%x
                vn=particles(cdx)%v
            do pdx=1,CHUNKSIZE
                xx=xn(pdx)
                Exn=0.0
                !$OMP SIMD reduction(+:Exn) private(psin)
                do kdx=1,Nf   
                    psin=exp( CMPLX(0.0, xx*kx0*kdx  ) )
                    Exn=Exn+2.0*(real(psin)*real(E(kdx)) - imag(psin)*imag(E(kdx)))
                end do
                vn(pdx)=vn(pdx)+dt*rksc(rkdx)*Exn
                xn(pdx)=xn(pdx)+dt*rksd(rkdx)*vn(pdx)
            end do
            particles(cdx)%v=vn
            particles(cdx)%x=xn

        end do
        !$OMP END PARALLEL DO
        
        
        
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





