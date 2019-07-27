/**
 * @file IO.cpp
 * @author Liran Nachman (lirannh@gmail.com)
 * @brief 
 * @version 0.1
 * @date 2019-07-27
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#define _CRT_SECURE_NO_WARNINGS
#include "IO.h"
int readFromFile(char * path, Point ** pts, int * n, int * k,
	double * alfa_zero, double * alfa_max, int * limit, double * QC)
{

	FILE* file = fopen(path, "r");
	if (file == NULL)
	{
		fprintf(stderr, "\nError opening file\n");
		return 0;
	}

	fscanf(file, "%d %d %lf %lf %d %lf", n, k, alfa_zero, alfa_max, limit, QC);
	fflush(NULL);

	printf("n = %d  k= %d alfa0 = %lf alfa_max = %lf limit = %d  QC =%lf \n", *n, *k, *alfa_zero, *alfa_max, *limit, *QC);
	fflush(NULL);

	*pts = (Point*)malloc(sizeof(Point) * (*n)); // create n array of points

	for (int i = 0; i < *n; i++)
	{
		(*pts)[i].values = (double*)malloc(sizeof(double)*(*k + 1)); // (2,20) exmaple

		for (int j = 0; j < *k; j++)
		{
			fscanf(file, "%lf", &(*pts)[i].values[j]);

		}
		(*pts)[i].values[*k] = 1; // plus 1 in points xi ( 2,3 ,1 )

		fscanf(file, "%d", &(*pts)[i].group);
	}

	fclose(file);

	printf("finish reading from file \n");
	fflush(NULL);

	return 1;

}

