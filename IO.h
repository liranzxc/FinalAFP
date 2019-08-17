/**
 * @file IO.h
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

/**
 * @read from file function
 * 
 * @param path of file name
 * @param pts pointer of pointer array of points size n
 * @param n 
 * @param k 
 * @param alfa_zero 
 * @param alfa_max 
 * @param limit 
 * @param QC 
 * @return int - status if read from file successfully
 */
int readFromFile(char * path, Point ** pts, int * n, int * k,
	double * alfa_zero, double * alfa_max, int * limit, double * QC);