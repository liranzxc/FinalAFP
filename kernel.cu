
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>

#include <stdio.h>
#include "kernel.h"
#include <math.h>
#include <omp.h>
#include <stdarg.h>


__device__ double dot(double * dev_w, double * dev_x,int indexValues, int * dev_k)
{
	double sum = 0;
	for (int i = 0; i < *dev_k + 1; i++)
		sum += dev_w[i] * dev_x[i+ indexValues];
	return sum;
}

__global__ void createNewWeight(double * dev_alfa, double *dev_values,int * indexerValues, double * W_dev)
{
	int i = threadIdx.x;
	W_dev[i] = (*dev_alfa)*dev_values[*indexerValues + i] + W_dev[i];
}

__global__ void	getMisLeadArrayFromPoints(Point * dev_pts, double* dev_values ,double * dev_W,
	int * dev_mislead, int * dev_k,int * dev_n) {

	int i = blockIdx.x * 1000 + threadIdx.x;
	if (i < *dev_n)
	{
		int indexValues = i *(*dev_k + 1);
		// calaculate fx 
		double fx = dot(dev_W, dev_values,indexValues, dev_k);
		int sign = fx >= 0 ? 1 : -1;
		if (dev_pts[i].group != sign)  {  // A group ,mislead
			sign = (dev_pts[i].group - sign) / 2;
			dev_mislead[i] = sign;
		}
		else
			dev_mislead[i] = 0;
	}
}



