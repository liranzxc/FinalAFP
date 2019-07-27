/**
 * @file cudaUtils.h
 * @author Liran Nachman (lirannh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2019-07-27
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#pragma once
#include "util.h"
#include "kernel.h"

/**
 * @ a cuda malloc helper 
 * @param dev_pointer 
 * @param size 
 * @param error_label 
 */
void MyCudaMalloc(void** dev_pointer, size_t size, int error_label);

/**
 * @ a cuda copy helper
 * @param dest 
 * @param src 
 * @param size 
 * @param kind 
 * @param error_label 
 */
void MyCudaCopy(void* dest, void * src, size_t size, cudaMemcpyKind kind, int error_label);


/**
 * @a free constanst cuda function will free all resources that const on GPU
 * @param dev_pts 
 * @param dev_values 
 * @param dev_n 
 * @param dev_k 
 */
void FreeConstanstCuda(Point * dev_pts, double * dev_values, int * dev_n, int * dev_k);

/**
 * @ malloc const paramaters on GPU
 * @param pts 
 * @param n 
 * @param k 
 * @param dev_pts 
 * @param dev_n 
 * @param dev_k 
 * @param dev_values 
 */
void mallocConstCuda(Point * pts, int n, int k, Point ** dev_pts, int ** dev_n, int ** dev_k, double ** dev_values);

/**
 * @ free cuda function helper 
 * @param object 
 * @param error_label 
 */
void MyCudaFree(void * object, int error_label);

/**
 * @free function wil free all resource that dynamic on process in the GPU
 * @param dev_W 
 * @param dev_alfa 
 * @param dev_mislead 
 * @return cudaError_t 
 */
cudaError_t FreeFunction(double * dev_W, double  * dev_alfa, int * dev_mislead);

/**
 * @process alfa function
 *  the function get alfa to process and run the algoritam
 * if the alfa good enough , return the q that match the alfa 
 * else return q = 2.0
 * also the Wsaved save the weight of vector in the end of algoritam
 * @param dev_pts 
 * @param dev_values 
 * @param alfa 
 * @param dev_n 
 * @param dev_k 
 * @param limit 
 * @param QC 
 * @param n 
 * @param k 
 * @param WSaved 
 * @return double 
 */
double ProcessAlfa(Point * dev_pts, double* dev_values, double  * alfa, int *dev_n
	, int *dev_k, int limit, double QC, int n, int k, double ** WSaved);