#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.h"


__device__ double dot(double * dev_w, double * dev_x, int indexValues, int * dev_k);

__global__ void createNewWeight(double * dev_alfa, double *dev_values, int * indexerValues, double * W_dev);

__global__ void	getMisLeadArrayFromPoints(Point * dev_pts, double* dev_values, double * dev_W, int * dev_mislead, int * dev_k, int * dev_n);