#pragma once

#include "util.h"
#include "kernel.h"

void MyCudaMalloc(void** dev_pointer, size_t size, int error_label);

void MyCudaCopy(void* dest, void * src, size_t size, cudaMemcpyKind kind, int error_label);

void FreeConstanstCuda(Point * dev_pts, double * dev_values, int * dev_n, int * dev_k);

void mallocConstCuda(Point * pts, int n, int k, Point ** dev_pts, int ** dev_n, int ** dev_k, double ** dev_values);

void MyCudaFree(void * object, int error_label);

cudaError_t FreeFunction(double * dev_W, double  * dev_alfa, int * dev_mislead, int * dev_tempresult);

double ProcessAlfa(Point * dev_pts, double* dev_values, double  * alfa, int *dev_n
	, int *dev_k, int limit, double QC, int n, int k, double ** WSaved);