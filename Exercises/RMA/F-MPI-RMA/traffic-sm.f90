program traffic

  use iso_c_binding, only: c_ptr, c_f_pointer

  use mpi

  use trafficlib

  implicit none

! Set the size of the road

  integer :: ncell = 100000
  integer :: maxiter, printfreq

  integer :: rank, size, ierr, nlocal, rankup, rankdown, tag = 1
  integer, dimension(MPI_STATUS_SIZE) :: status

  integer :: i, iter, nmove, nmovelocal, ncars
  real    :: density

  integer, allocatable, dimension(:) :: newroad, bigroad
  integer, pointer, dimension(:) :: oldroadtmp, oldroad

  double precision :: tstart, tstop

  ! RMA variables

  integer :: nodecomm
  integer :: nodesize, noderank, nodestringlen
  integer :: nodewin
  integer(MPI_ADDRESS_KIND) :: winsize
  integer :: intsize, disp_unit
  character*(MPI_MAX_PROCESSOR_NAME) :: nodename
  type(c_ptr) :: baseptr

  
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
!  allocate(oldroad(0:nlocal+1))

  ! Create node-local communicator

  call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, &
                           MPI_INFO_NULL, nodecomm, ierr)

  ! Check it all went as expected
  
  call MPI_Get_processor_name(nodename, nodestringlen, ierr)
  call MPI_Comm_size(nodecomm, nodesize, ierr)
  call MPI_Comm_rank(nodecomm, noderank, ierr)

  write(*,*) 'Rank ', rank,' in COMM_WORLD is rank ', noderank, &
             ' in nodecomm on node ', nodename(1:nodestringlen)

  call MPI_Type_size(MPI_INTEGER, intsize, ierr)

  ! Allocated oldroad as a shared array, contiguous across processes

  winsize = (nlocal+2)*intsize

  ! displacements counted in units of integers

  disp_unit = intsize

  call MPI_Win_allocate_shared(winsize, disp_unit, &
       MPI_INFO_NULL, nodecomm, baseptr, nodewin, ierr)

  ! coerce baseptr to a Fortran array

  call c_f_pointer(baseptr, oldroadtmp, [nlocal+2])

  ! make sure indexing starts at 0 not at 1

  oldroad(0:nlocal+1) => oldroadtmp(1:nlocal+2)
  
  do i = 1, nlocal
     oldroad(i) = 0
     newroad(i) = 0
  end do

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

     ! Use direct memory write for on-node comms

     ! Ensure everyone is ready
     
     call MPI_Win_fence(0, nodewin, ierr)

     ! Write last cell (entry nlocal) to lower halo (entry 0)
     ! of up neighbour which is entry nlocal+2 of local array

     if (noderank /= nodesize-1) then

        oldroad(nlocal+2) = oldroad(nlocal)

     end if

     ! Write first cell (entry 1) to upper halo (entry nlocal+1)
     ! of down neighbour which is entry -1 of local array

     if (noderank /= 0) then

        oldroad(-1) = oldroad(1)

     end if

     ! Ensure everyone is finished before accessing oldroad

     call MPI_Win_fence(0, nodewin, ierr)

     ! Now do standard MPI communications extreme end points

     if (noderank == nodesize-1) then

        ! Send upper boundary to and receive upper halo from rankup

        call MPI_Ssend(oldroad(nlocal),  1, MPI_INTEGER, rankup, tag, &
		       MPI_COMM_WORLD, ierr)

        call MPI_Recv(oldroad(nlocal+1), 1, MPI_INTEGER, rankup, tag, &
                      MPI_COMM_WORLD, status, ierr)
     end if

     if (noderank == 0) then

        ! Send lower boundary to and receive lower halo from rankdown

        call MPI_Recv(oldroad(0), 1, MPI_INTEGER, rankdown, tag, &
                      MPI_COMM_WORLD, status, ierr)

	call MPI_Ssend(oldroad(1), 1, MPI_INTEGER, rankdown, tag, &
                       MPI_COMM_WORLD, ierr)

      end if



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
