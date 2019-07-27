/**
 * @file kernel.h
 * @author Liran Nachman (lirannh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2019-07-27
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "util.h"

/**
 * @scalar product is an algebraic operation 
 * that takes two equal-length sequences of numbers 
 * (usually coordinate vectors) and returns a single number
 * 
 * @param dev_w - vector of weights in size k+1
 * @param dev_x  - vector of coordinates in size k+1
 * @param indexValues - a index of entry values in dev_x
 * @param dev_k 
 * @return double - results 
 */
__device__ double dot(double * dev_w, double * dev_x, int indexValues, int * dev_k);

/**
 * @create a new weight vector 
 * @param dev_alfa 
 * @param dev_values - vector of coordinates
 * @param indexerValues - a index of entry values in dev_values
 * @param W_dev - old weight vector 
 * @return void - set on W_dev the new weight vector
 */
__global__ void createNewWeight(double * dev_alfa, double *dev_values, int * indexerValues, double * W_dev);


/**
 * @ get MisLead Array From Points
 * the function check if point is in the currectly place,if yes set 0 on results ,else 
 * set the sign of fx ( 1 or -1) 
 * @param dev_pts - gpu array of points 
 * @param dev_values - gpu array of values of points 
 * @param dev_W  - gpu weight vector
 * @param dev_mislead - array of save the results
 * @param dev_k - k dims
 * @param dev_n  - n points
 * @return void - set on dev_mislead a array of results 
 */
__global__ void	getMisLeadArrayFromPoints(Point * dev_pts, double* dev_values, double * dev_W, int * dev_mislead, int * dev_k, int * dev_n);