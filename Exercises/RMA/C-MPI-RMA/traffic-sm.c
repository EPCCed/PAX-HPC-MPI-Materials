#include <stdlib.h>
#include <stdio.h>

#include <mpi.h>

#include "traffic.h"

int main(int argc, char **argv)
{
  // Set the size of the road

  int ncell = 100000;

  int *oldroad, *newroad, *bigroad;

  int i, iter, nmove, nmovelocal, ncars;
  int maxiter, printfreq; 

  float density; 

  double tstart, tstop;

  MPI_Status status;
  int rank, size, nlocal, rankup, rankdown;
  int tag = 1;

  // RMA variables

  MPI_Comm nodecomm;
  int nodesize, noderank, nodestringlen, upoffset, dnoffset;
  MPI_Win nodewin;
  MPI_Aint winsize;
  int disp_unit;
  char nodename[MPI_MAX_PROCESSOR_NAME];

  maxiter = 200000000/ncell; 
  printfreq = maxiter/10; 

  // Set target density of cars

  density = 0.52;

  printf("Running message-passing traffic model\n");

  // Start MPI

  MPI_Init(NULL, NULL);  

  // Find size and rank

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size == 1)
    {
      printf("Error: naive method for MPI halo-swapping requires more than 1 process\n");
      MPI_Finalize();
      return 1;
    }

  if (rank == 0)
    {
      printf("\nLength of road is %d\n", ncell);
      printf("Number of iterations is %d \n", maxiter);
      printf("Target density of cars is %f \n", density);
      printf("Running on %d processes\n", size);
    }

  nlocal = ncell/size;

  // Create node-local communicator

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
              MPI_INFO_NULL, &nodecomm);

  // Check it all went as expected
  
  MPI_Get_processor_name(nodename, &nodestringlen);
  MPI_Comm_size(nodecomm, &nodesize);
  MPI_Comm_rank(nodecomm, &noderank);

  printf("Rank %d in COMM_WORLD is rank %d in nodecomm on node <%s>\n",
	 rank, noderank, nodename);

  //  oldroad = (int *) malloc((nlocal+2)*sizeof(int));
  newroad = (int *) malloc((nlocal+2)*sizeof(int));

  // Allocated oldroad as a shared array, contiguous across processes

  winsize = (nlocal+2)*sizeof(int);

  // displacements counted in units of integers

  disp_unit = sizeof(int);

  MPI_Win_allocate_shared(winsize, disp_unit,
              MPI_INFO_NULL, nodecomm, &oldroad, &nodewin);

  for (i=1; i <= nlocal; i++)
    {
      oldroad[i] = 0;
      newroad[i] = 0;
    }

  if (rank == 0)
    {
      bigroad = (int *) malloc(ncell*sizeof(int));

      // Initialise road accordingly using random number generator

      printf("Initialising road ...\n");
  
      ncars = initroad(bigroad, ncell, density, SEED);

      printf("...done\n");
      printf("Actual density is %f\n", (float) ncars / (float) ncell);
      printf("Scattering data ...\n");
    }

  MPI_Scatter(bigroad,     nlocal, MPI_INT,
	      &oldroad[1], nlocal, MPI_INT,
	      0, MPI_COMM_WORLD);

  if (rank == 0)
    {
      printf("... done\n\n");
    }

  // Compute neighbours

  rankup   = (rank + 1) % size;
  rankdown = (rank + size - 1) % size;

  tstart = MPI_Wtime();

  for (iter=1; iter<=maxiter; iter++)
    { 
      // Implement halo swaps which now includes boundary conditions

      // Use direct memory write for on-node comms

      // Ensure everyone is ready

      MPI_Win_fence(0, nodewin);

      // Write last cell (entry nlocal) to lower halo (entry 0)
      // of up neighbour which is entry nlocal+2 of local array

      if (noderank != nodesize-1)
	{
	  oldroad[nlocal+2] = oldroad[nlocal];
	}

      // Write first cell (entry 1) to upper halo (entry nlocal+1)
      // of down neighbour which is entry -1 of local array

      if (noderank != 0)
	{
	  oldroad[-1] = oldroad[1];
	}

      // Ensure everyone is finished before accessing oldroad

      MPI_Win_fence(0, nodewin);

      // Now do standard MPI communications extreme end points

      if (noderank == nodesize-1)
	{
	  // Send upper boundary to and receive upper halo from rankup

	  MPI_Ssend(&oldroad[nlocal],  1, MPI_INT, rankup, tag,
		   MPI_COMM_WORLD);

	  MPI_Recv(&oldroad[nlocal+1], 1, MPI_INT, rankup, tag,
		    MPI_COMM_WORLD, &status);
	}

      if (noderank == 0)
	{
	  // Send lower boundary to and receive lower halo from rankdown

	  MPI_Recv(&oldroad[0],     1, MPI_INT, rankdown, tag,
		    MPI_COMM_WORLD, &status);

	  MPI_Ssend(&oldroad[1],    1, MPI_INT, rankdown, tag,
		   MPI_COMM_WORLD);
	}

      // Apply CA rules to all cells

      nmovelocal = updateroad(newroad, oldroad, nlocal);

      // Globally sum the value

      MPI_Allreduce(&nmovelocal, &nmove, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

      // Copy new to old array

      for (i=1; i<=nlocal; i++)
	{
	  oldroad[i] = newroad[i]; 
	}

      if (iter%printfreq == 0)
	{
	  if (rank == 0)
	    {
	      printf("At iteration %d average velocity is %f \n",
		     iter, (float) nmove / (float) ncars);
	    }
	} 
    }

  tstop = MPI_Wtime();

  //  free(oldroad);

  MPI_Win_free(&nodewin);
  free(newroad);

  if (rank == 0)
    {
      free(bigroad);

      printf("\nFinished\n");
      printf("\nTime taken was  %f seconds\n", tstop-tstart);
      printf("Update rate was %f MCOPs\n\n", \
      1.e-6*((double) ncell)*((double) maxiter)/(tstop-tstart));
    }

  // Finish

  MPI_Finalize();
}
