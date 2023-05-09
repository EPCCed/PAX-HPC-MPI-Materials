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

  MPI_Win win;
  MPI_Aint winsize;
  int disp_unit;

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

  if (rank == 0)
    {
      printf("\nLength of road is %d\n", ncell);
      printf("Number of iterations is %d \n", maxiter);
      printf("Target density of cars is %f \n", density);
      printf("Running on %d processes\n", size);
    }

  nlocal = ncell/size;

  oldroad = (int *) malloc((nlocal+2)*sizeof(int));
  newroad = (int *) malloc((nlocal+2)*sizeof(int));

  // Publish oldroad for RMA operations

  winsize = (nlocal+2)*sizeof(int);

  // displacements counted in units of integers

  disp_unit = sizeof(int);

  MPI_Win_create(oldroad, winsize, disp_unit,
		 MPI_INFO_NULL, MPI_COMM_WORLD, &win);
  
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

      /*      MPI_Sendrecv(&oldroad[nlocal], 1, MPI_INT, rankup, tag,
		   &oldroad[0],      1, MPI_INT, rankdown, tag,
		   MPI_COMM_WORLD, &status);

      MPI_Sendrecv(&oldroad[1],        1, MPI_INT, rankdown, tag,
		   &oldroad[nlocal+1], 1, MPI_INT, rankup, tag,
		   MPI_COMM_WORLD, &status); */


      // Implement halo swaps using RMA puts

      // Ensure everyone is ready

      MPI_Win_fence(0, win);

      // Write last cell (entry nlocal) to lower halo (entry 0)
      // of up neighbour (rankip)

      MPI_Put(&oldroad[nlocal], 1, MPI_INT, rankup,
	      0,                1, MPI_INT, win);

      // Write first cell (entry 1) to upper halo (entry nlocal+1)
      // of down neighbour (rankdown)

      MPI_Put(&oldroad[1], 1, MPI_INT, rankdown,
	      nlocal+1,    1, MPI_INT, win);


      // Ensure everyone is finished before accessing oldroad

      MPI_Win_fence(0, win);


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

  free(oldroad);
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
