/**
 * @file program.cpp
 * @author Liran Nachman (lirannh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2019-07-27
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#include <omp.h>
#include "mpi.h"
#include "IO.h"
#include "cudaUtils.h"
#include "util.h"


int main(int argc, char *argv[])
{

#pragma region mpi startup 

	int myrank, size;
	MPI_Status status;
	double exit_label = -1;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);


	int n, k, limit;
	double alfa_zero, alfa_max, QC;
	struct Point * pts;

#pragma endregion

#pragma region master read from file

	if (myrank == MASTER)
	{
		if (argc != 2)
		{
			printf("ERROR : error must be file path \n");
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}
		char * path = argv[1]; // file path
		if (readFromFile(path, &pts, &n, &k, &alfa_zero, &alfa_max, &limit, &QC) == 0) {
			printf(" ERROR : error reading from file \n");
			MPI_Abort(MPI_COMM_WORLD, 1);
			exit(1);
		}

		printf("master finish reading from file \n");
		fflush(NULL);
	}

#pragma endregion

#pragma region master_boardcast status - all processes have same data without alfas

		// bcast all information
		MPI_Bcast(&k, 1, MPI_INT, MASTER, MPI_COMM_WORLD); // for dim type point
		MPI_Bcast(&limit, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&QC, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
		MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);


	//	 allocation points
		if (myrank != MASTER)
		{
			pts = (Point*)malloc(sizeof(Point) * (n)); // create n array of points
			#pragma omp parallel for
			for (int i = 0; i < n; i++)
			{
				pts[i].values = (double*)malloc(sizeof(double)*(k + 1)); // (2,20) exmaple
			}
		}
		//boardcast all values
		for (int i = 0; i < n; i++)
		{
			MPI_Bcast(&pts[i].values[0], k + 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
			MPI_Bcast(&pts[i].group, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
		}
#pragma  endregion 

#pragma region init cuda const
		// init cuda constact 
		Point * dev_pts = NULL;
		int * dev_n = NULL;
		int * dev_k = NULL;
		double * dev_values = NULL;

		double * qMin = (double*)malloc(sizeof(double));
		double * wMin = (double*)malloc(sizeof(double)*(k+1));
		double * alfaMin = (double*)malloc(sizeof(double));

		// setup coda constast
		mallocConstCuda(pts, n, k, &dev_pts, &dev_n, &dev_k, &dev_values);
		
#pragma endregion

#pragma region work dynamic 
		// master section
		if (myrank == MASTER)
		{

			double * AlfasFromProcess = (double*)malloc(sizeof(double)*size);
			double currectWorkAlfa = alfa_zero;
			int currectwork = 1;
			double miniumOfminimusAlfas;
			double indexProcessofMinimunAlfa = -1;
			double minQFound;
			double * minWFound = (double*)malloc(sizeof(double)*k + 1);

			// start send dynamic alfa
			while (currectWorkAlfa < alfa_max) // dynmaic work
			{
				double startAlfaWorker = currectWorkAlfa;
				currectWorkAlfa = currectWorkAlfa + alfa_zero*NUM_ALFA_TO_SEND_EACH_PROCESS;
				if (currectWorkAlfa > alfa_max) currectWorkAlfa = alfa_max;
				

				// each worker get interval of alfa and alfa step (alfa zero)
				MPI_Send(&startAlfaWorker, 1, MPI_DOUBLE, currectwork, 0, MPI_COMM_WORLD);
				MPI_Send(&currectWorkAlfa, 1, MPI_DOUBLE, currectwork, 0, MPI_COMM_WORLD);
				MPI_Send(&alfa_zero, 1, MPI_DOUBLE, currectwork, 0, MPI_COMM_WORLD);

				currectwork = ((currectwork + 1) % size);

				if (currectwork == MASTER && currectWorkAlfa < alfa_max)
					// finish one sending data cycle ,so master do job
			    {
					startAlfaWorker = currectWorkAlfa;
					currectWorkAlfa = currectWorkAlfa + alfa_zero*NUM_ALFA_TO_SEND_EACH_PROCESS;
					if (currectWorkAlfa > alfa_max) currectWorkAlfa = alfa_max;
			
					// do job 
					DoJob(startAlfaWorker, currectWorkAlfa,alfa_zero, dev_pts, dev_values, dev_n, dev_k, n, k, limit, QC, qMin, wMin, alfaMin);

					if (*qMin == 2.0)
					{
						AlfasFromProcess[MASTER] = -1;
					}
					else
					{
						AlfasFromProcess[MASTER] = *alfaMin;

					}

					printf("start recive from process \n ");
					fflush(NULL);
					int counterRecive = 1;
					double tempResultAlfa = 0;
					while (counterRecive < size)
					{
						MPI_Recv(&tempResultAlfa, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
						AlfasFromProcess[status.MPI_SOURCE] = tempResultAlfa;
						counterRecive += 1;
					}

					//receive from all need to find minimum
					miniumOfminimusAlfas = AlfasFromProcess[0];
					for (int i = 1; i < size; i++)
					{
						if (AlfasFromProcess[i] == -1)
							continue;
						else
						{
							if (miniumOfminimusAlfas > AlfasFromProcess[i])
							{
								miniumOfminimusAlfas = AlfasFromProcess[i];
								indexProcessofMinimunAlfa = i;
							}
						}
					}
			

					if (miniumOfminimusAlfas != -1) // we found some alfa
					{ 

						int statusSuccfully = 200;
						// get all information for process that have alfa 
						MPI_Send(&statusSuccfully, 1, MPI_INT, indexProcessofMinimunAlfa, 0, MPI_COMM_WORLD);
						MPI_Recv(&minQFound, 1, MPI_DOUBLE, indexProcessofMinimunAlfa, 0, MPI_COMM_WORLD, &status);
						MPI_Recv(minWFound, k+1, MPI_DOUBLE, indexProcessofMinimunAlfa, 0, MPI_COMM_WORLD, &status);
						break;
					}

					//we dont found something , send to next worker data to work on.
					currectwork = ((currectwork + 1) % size);
				}
			}


			if (miniumOfminimusAlfas != -1)
			{

				printf("Found results : \n");
				fflush(NULL);

				printf("alfa %f , q = %lf , from process %d \n", miniumOfminimusAlfas, minQFound, myrank);
				fflush(NULL);

				printf("mini value of W : [");
				for (int i = 0; i < k + 1; i++)
				{
					printf("%f,", minWFound[i]);
					fflush(NULL);


				}
				printf("] \n");
				fflush(NULL);
			}
			else
			{
				printf("Alfa not found !");
				fflush(NULL);

			}

			for (int worker = 0; worker < size; worker++) // terminal all workers
			{
				MPI_Send(&exit_label, 1, MPI_DOUBLE, currectwork, 0, MPI_COMM_WORLD);
				MPI_Send(&exit_label, 1, MPI_DOUBLE, currectwork, 0, MPI_COMM_WORLD);
				MPI_Send(&exit_label, 1, MPI_DOUBLE, currectwork, 0, MPI_COMM_WORLD);

			}
		} 
		// worker section 
		else // workers
		{
			double step;
			int statusContinue = 0;
			while (true) // slaves get work
			{
				MPI_Recv(&alfa_zero, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&alfa_max, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);
				MPI_Recv(&step, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, &status);


				if (alfa_zero == exit_label || alfa_max == exit_label || step == exit_label)
				{
					break;
				}

		
				// worker do job
				DoJob(alfa_zero, alfa_max,step, dev_pts, dev_values, dev_n, dev_k, n, k, limit, QC, qMin, wMin, alfaMin);
			

				if (*qMin == 2.0)
				{
					// not found
					double ALFA_NOT_FOUND = -1;
					MPI_Send(&ALFA_NOT_FOUND, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);

				}
				else
				{ 
					// worker found good alfa , send alfa and wait for status 
					MPI_Send(alfaMin, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
					MPI_Recv(&statusContinue, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status); // wait for status

					if (statusContinue == 200) // i am the minium  !
					{
						MPI_Send(qMin, 1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
						MPI_Send(wMin, k+1, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);

					}
					else if (statusContinue == exit_label) // some process have a minimun alfa then I
						break;
				}
			}
		}


#pragma endregion
	
		
// finalize all process
MPI_Finalize();
return 0;
}