/**
 * @file util.cpp
 * @author Liran Nachman (lirannh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2019-07-27
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "util.h"
#include "cudaUtils.h"


void DoJob(double alfaZero, double alfaMax, double stepAlfa, Point * dev_pts, double * dev_values, int * dev_n, int * dev_k, int n, int k, int limit,
	double QC, double * qMin, double * wMin, double *alfaMin)
{
	double maxIteraction = (alfaMax - alfaZero) / stepAlfa;
	maxIteraction = round(maxIteraction);
	int numofSteps = (int)maxIteraction;
	
	if (numofSteps == 0)
	{
		*qMin = 2.0;
		return;
	}
	// create for each alfa W to save 
	double ** WMinForArr = (double**)malloc(numofSteps * sizeof(double *));

	// get minimun 
	double qMinFor = 2.0;
	int indexerMinFor = -1;
	double alfaMinFor = -1;
// openMP loop over alfa
#pragma omp parallel for
	for (int i = 0; i < numofSteps; i++) // running over all alfa
	{
		double * tempAlfa = (double*)malloc(sizeof(double));
		*tempAlfa = alfaZero + i*alfaZero;
		double tempQ = ProcessAlfa(dev_pts, dev_values, tempAlfa, dev_n, dev_k, limit, QC, n, k, &WMinForArr[i]);
	
		#pragma omp critical
		{
			if (tempQ < qMinFor)
			{
				qMinFor = tempQ;
				indexerMinFor = i;
				alfaMinFor = alfaZero + i*alfaZero;
			}
		}
	}

#pragma region save pointers values
	*qMin = qMinFor;
	*alfaMin = alfaMinFor;
	for (int i = 0; i < k + 1; i++)
	{
		wMin[i] = WMinForArr[indexerMinFor][i];
	}

#pragma endregion

#pragma region Free resources
	for (int i = 0; i < numofSteps; i++)
	{
		free(WMinForArr[i]);
	}
	free(WMinForArr);
	if (qMinFor != 2.0) 
		// two options , I am the process that send Q , or some process have lower q then me 
		// each options we stop the algoritam ,as results we free pointers.
	{
		FreeConstanstCuda(dev_pts, dev_values, dev_n, dev_k);
	}
#pragma endregion
}