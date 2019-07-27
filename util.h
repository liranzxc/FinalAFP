#pragma once


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <stdexcept>
#define MASTER 0
#define NUM_ALFA_TO_SEND_EACH_PROCESS 1
struct Point
{
	int group;
	double * values;
};


//doJob function , each process will execute that function to calcaulate the alfa's.
// the function will return a minium W , and minium q ,miniumalfa
// if all alfas dont good enough for QC, return q = 2 
void DoJob(double alfaZero, double alfaMax,double step,
	Point * dev_pts, double * dev_values, int * dev_n, int * dev_k, int n, int k, int limit,
	double QC, double * qMin, double * wMin, double *alfaMin);