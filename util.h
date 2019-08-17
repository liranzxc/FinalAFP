/**
 * @file util.h
 * @author Liran Nachman (lirannh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2019-07-27
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdarg.h>
#include <stdexcept>
#define MASTER 0
#define NUM_ALFA_TO_SEND_EACH_PROCESS 4
<<<<<<< HEAD
=======

/**
 * point struct 
 * int group - assign group of point
 * double values - a coordinates array of points size K+1 
 */ 
>>>>>>> 2946dfd6e9f6b66b9dcec86932d2d13e12e48bb9
struct Point
{
	int group;
	double * values;
};

/**
 * @ do job function   
 * input : a interval of alfas 
 * output : set the minimun of alfa that good enough q< QC
 * if none of alfas was enough set q = 2.0
 * @param alfaZero 
 * @param alfaMax 
 * @param step 
 * @param dev_pts 
 * @param dev_values 
 * @param dev_n 
 * @param dev_k 
 * @param n 
 * @param k 
 * @param limit 
 * @param QC 
 * @param qMin 
 * @param wMin 
 * @param alfaMin 
 */
void DoJob(double alfaZero, double alfaMax,double step,
	Point * dev_pts, double * dev_values, int * dev_n, int * dev_k, int n, int k, int limit,
	double QC, double * qMin, double * wMin, double *alfaMin);