program traffic

  use mpi

  use trafficlib

  implicit none

! Set the size of the road

  integer :: ncell = 100000
  integer :: maxiter, printfreq

  integer rank, size, ierr, nlocal, rankup, rankdown
  integer, dimension(MPI_STATUS_SIZE) :: status

  integer :: i, iter, nmove, nmovelocal, ncars
  real    :: density

  integer, allocatable, dimension(:) :: newroad, oldroad, bigroad

  double precision :: tstart, tstop

  ! RMA variables

  integer :: win
  integer(MPI_ADDRESS_KIND) :: winsize, target_disp
  integer :: intsize, disp_unit

  maxiter = 200000000/ncell
  printfreq = maxiter/10

! Set target density of cars

  density = 0.52

  write(*,*) 'Running message-passing traffic model'

! Start message passing system

  call MPI_Init(ierr)

! Compute size and rank

  call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

  nlocal = ncell/size

  if (rank == 0) then

     write(*,*)
     write(*,*) 'Length of road is ', ncell
     write(*,*) 'Number of iterations is ', maxiter
     write(*,*) 'Target density of cars is ', density
     write(*,*) 'Running on ', size, ' processes'

  end if

! Allocate arrays

  allocate(newroad(0:nlocal+1))
  allocate(oldroad(0:nlocal+1))

  do i = 1, nlocal
     oldroad(i) = 0
     newroad(i) = 0
  end do

  ! Publish oldroad for RMA operations

  call MPI_Type_size(MPI_INTEGER, intsize, ierr)

  winsize = (nlocal+2)*intsize;

  ! displacements counted in units of integers

  disp_unit = intsize;

  call MPI_Win_create(oldroad, winsize, disp_unit, &
                      MPI_INFO_NULL, MPI_COMM_WORLD, win, ierr)
  
  if (rank == 0) then

     allocate(bigroad(ncell))

     ! Initialise road accordingly using random number generator

     write(*,*) 'Initialising ...'

     ncars = initroad(bigroad, ncell, density, seed)

     write(*,*) '... done'

     write(*,*) 'Actual density of cars is ', float(ncars)/float(ncell)
     write(*,*) 'Scattering data ...'

  end if

  call MPI_Scatter(bigroad,    nlocal, MPI_INTEGER, &
                   oldroad(1), nlocal, MPI_INTEGER, &
                   0, MPI_COMM_WORLD, ierr)

  if (rank == 0) then

     write(*,*) '... done'
     write(*,*)

  end if

! Compute neighbours

  rankup   = mod(rank + 1, size)
  rankdown = mod(rank + size - 1, size)

  tstart = MPI_Wtime()

  do iter = 1, maxiter

! Implement halo swaps which now includes boundary conditions

!     call MPI_Sendrecv(oldroad(nlocal), 1, MPI_INTEGER, rankup, 1,   &
!                       oldroad(0),      1, MPI_INTEGER, rankdown, 1, &
!                       MPI_COMM_WORLD, status, ierr);

!     call MPI_Sendrecv(oldroad(1),        1, MPI_INTEGER, rankdown, 1,  &
!                       oldroad(nlocal+1), 1, MPI_INTEGER, rankup, 1,    &
!                       MPI_COMM_WORLD, status, ierr);

     ! Implement halo swaps using RMA puts

     ! Ensure everyone is ready

     call MPI_Win_fence(0, win, ierr)

     ! Write last cell (entry nlocal) to lower halo (entry 0)
     ! of up neighbour (rankup)

     target_disp = 0

     call MPI_Put(oldroad(nlocal), 1, MPI_INTEGER, rankup, &
                  target_disp,     1, MPI_INTEGER, win, ierr)

     ! Write first cell (entry 1) to upper halo (entry nlocal+1)
     ! of down neighbour (rankdown)

     target_disp = nlocal+1

     call MPI_Put(oldroad(1),  1, MPI_INTEGER, rankdown, &
	          target_disp, 1, MPI_INTEGER, win, ierr)


     ! Ensure everyone is finished before accessing oldroad

     call MPI_Win_fence(0, win, ierr)

     nmovelocal = updateroad(newroad, oldroad, nlocal)

! Globally sum the value

     call MPI_Allreduce(nmovelocal, nmove, 1, MPI_INTEGER, MPI_SUM, &
                        MPI_COMM_WORLD, ierr)

! Copy new to old array

     do i = 1, nlocal

        oldroad(i) = newroad(i)

     end do

     if (mod(iter, printfreq) == 0) then

        if (rank == 0) write(*,*) 'At iteration ', iter, &
             ' average velocity is ', float(nmove)/float(ncars)
     end if

  end do

  tstop = MPI_Wtime()
  
  deallocate(oldroad)
  deallocate(newroad)

  if (rank == 0) then

     deallocate(bigroad)

     write(*,*)
     write(*,*) 'Finished'
     write(*,*)
     write(*,*) 'Time taken was  ', tstop - tstart, ' seconds'
     write(*,*) 'Update rate was ', &
                 1.d-6*float(ncell)*float(maxiter)/(tstop-tstart), ' MCOPs'
     write(*,*)

  end if

  call MPI_Finalize(ierr)

end program traffic
