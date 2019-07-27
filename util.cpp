#include "util.h"
#include "cudaUtils.h"


void DoJob(double alfaZero, double alfaMax, double stepAlfa, Point * dev_pts, double * dev_values, int * dev_n, int * dev_k, int n, int k, int limit,
	double QC, double * qMin, double * wMin, double *alfaMin)
{
	double maxIteraction = (alfaMax - alfaZero) / stepAlfa;
	maxIteraction = round(maxIteraction);
	int numofSteps = (int)maxIteraction;

	// create for each alfa , q and W to save 
	double * QSaved = (double*)malloc(numofSteps * sizeof(double));
	double ** WSaved = (double**)malloc(numofSteps * sizeof(double *));

// openMP loop over alfa
#pragma omp parallel for
	for (int i = 0; i < numofSteps; i++) // running over all alfa
	{
		double * tempAlfa = (double*)malloc(sizeof(double));
		*tempAlfa = alfaZero + i*alfaZero;
		QSaved[i] = ProcessAlfa(dev_pts, dev_values, tempAlfa, dev_n, dev_k, limit, QC, n, k, &WSaved[i]);
	}

#pragma region get mininum q from all qs
	double minQ = QSaved[0];
	int indexer = 0;
	for (int i = 1; i < numofSteps; i++)
	{
		if (QSaved[i] < minQ)
		{
			minQ = QSaved[i];
			indexer = i;
		}
	}
#pragma endregion

#pragma region save pointers values
	double alfaDynamic = alfaZero + indexer *alfaZero;
	*qMin = minQ;
	*alfaMin = alfaDynamic;
	for (int i = 0; i < k + 1; i++)
	{
		wMin[i] = WSaved[indexer][i];
	}

#pragma endregion

#pragma region Free resources
	for (int i = 0; i < numofSteps; i++)
	{
		free(WSaved[i]);
	}
	free(QSaved);
	free(WSaved);

	if (minQ != 2.0) 
		// two options , I am the process that send Q , or some process have lower q then me 
		// each options we stop the algoritam ,as results we free pointers.
	{
		FreeConstanstCuda(dev_pts, dev_values, dev_n, dev_k);
	}

#pragma endregion
}