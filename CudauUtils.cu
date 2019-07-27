
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cudaUtils.h"
#include "kernel.h"

cudaError_t FreeFunction(double * dev_W, double * dev_alfa, int * dev_mislead)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaFree(dev_W);
	if (cudaStatus != cudaSuccess) 
		printf("failed to free cuda - W  \n");

	cudaStatus = cudaFree(dev_mislead);
	if (cudaStatus != cudaSuccess) 
		printf("failed to free cuda - mislead points \n");

	cudaStatus = cudaFree(dev_alfa);
	if (cudaStatus != cudaSuccess) 
		printf("failed to free cuda - alfa \n");


	return cudaStatus;
}

void MyCudaMalloc(void** dev_pointer, size_t size, int error_label)
{
	cudaError_t cudaStatus;

	// points malloc n dims  .
	cudaStatus = cudaMalloc(dev_pointer, size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed! error_label : %d ", error_label);
		MyCudaFree(*dev_pointer, error_label);
	}


}

void MyCudaCopy(void* dest, void * src, size_t size, cudaMemcpyKind kind, int error_label)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaMemcpy(dest, src, size, kind);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed! error_label : %d", error_label);
}

void MyCudaFree(void * object, int error_label)
{
	cudaError_t cudaStatus;
	cudaStatus = cudaFree(object);
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "cudaMemcpy failed! error_label : %d", error_label);
}

void FreeConstanstCuda(Point * dev_pts, double * dev_values, int * dev_n, int * dev_k)
{
	MyCudaFree(dev_pts, 03);
	MyCudaFree(dev_values, 04);
	MyCudaFree(dev_n, 05);
	MyCudaFree(dev_k, 06);

}

void mallocConstCuda(Point * pts, int n, int k, Point ** dev_pts, int ** dev_n, int ** dev_k, double ** dev_values)
{

	MyCudaMalloc((void**)&(*dev_pts), sizeof(Point)* n, 1);
	MyCudaMalloc((void**)&(*dev_values), sizeof(double)* (n*(k + 1)), 2); // value n * (k+1) each point have k+1 dims values
	MyCudaCopy((*dev_pts), pts, sizeof(Point)*n, cudaMemcpyHostToDevice, 4);

	for (int i = 0; i < n; i++)
		MyCudaCopy(&(*dev_values)[i*(k + 1)], &pts[i].values[0], sizeof(double)*(k + 1), cudaMemcpyHostToDevice, 5);


	MyCudaMalloc((void**)&(*dev_n), sizeof(int), 265);
	MyCudaMalloc((void**)&(*dev_k), sizeof(int), 3);
	MyCudaCopy((*dev_n), &n, 1, cudaMemcpyHostToDevice, 5);
	MyCudaCopy((*dev_k), &k, 1, cudaMemcpyHostToDevice, 6);


}

double ProcessAlfa(Point * dev_pts, double* dev_values, double  * alfa, int *dev_n,
	int *dev_k, int limit, double QC, int n, int k, double ** WSaved)
{
	*WSaved = (double*)malloc((k + 1) * sizeof(double)); // W k+1 dims 
	int * mislead = (int*)malloc(n * sizeof(int)); // array of n points , 
	//mislead points will be 1 or -1 ,currect=0
	int * dev_mislead = NULL;
	double * dev_W = NULL;
	double * dev_alfa = NULL;
	cudaError_t cudaStatus;

#pragma region malloc and copy values (W ,mislead , alfa,tempresult) to GPU


	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		FreeFunction(dev_W, dev_alfa, dev_mislead);
		return -2;
	}



	MyCudaMalloc((void**)&dev_W, sizeof(double)* (k + 1), 7);
	cudaMemset(dev_W, 0, sizeof(double)* (k + 1)); // calloc cuda


	MyCudaMalloc((void**)&dev_mislead, sizeof(int)* (n), 9);
	cudaMemset(dev_mislead, 0, sizeof(int)* (n));

	MyCudaMalloc((void**)&dev_alfa, sizeof(double), 11);
	MyCudaCopy(dev_alfa, alfa, sizeof(double), cudaMemcpyHostToDevice, 12);

	

	MyCudaCopy(dev_n, &n, sizeof(int), cudaMemcpyHostToDevice, 14);


#pragma  endregion


	int threadDims = 1000;
	int blockDims = (n / threadDims) + 1;
	int counter_limit = 0;

	while (counter_limit < limit)
	{
		// get all mislead points
		getMisLeadArrayFromPoints << <blockDims, threadDims >> > (dev_pts, dev_values, dev_W, dev_mislead, dev_k, dev_n);

		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "loopOverPoints2 launch failed: %s\n", cudaGetErrorString(cudaStatus));
			FreeFunction(dev_W, dev_alfa, dev_mislead);
			return -2;

		}
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching loopOverPoints! : %s \n", cudaStatus, cudaGetErrorString(cudaStatus));
			FreeFunction(dev_W, dev_alfa, dev_mislead);
			return -2;

		}

		// copy mislead points from gpu after calc FX for all points
		MyCudaCopy(mislead, dev_mislead, (n) * sizeof(int), cudaMemcpyDeviceToHost, 15);


		int indexerMiss = 0;
		int result;
		// we get a array of mislead values (0 == ok , else 1||-1 are mislead by sign ));
		for (indexerMiss = 0; indexerMiss < n; indexerMiss++)
		{
			result = mislead[indexerMiss];
			if (result == -1 || result == 1) // found point that mislead
				break;
		}

		if (indexerMiss == n)// all point in good places
			break;
		else
		{
			// need to create a new W 

			*alfa = *alfa*mislead[indexerMiss]; // alfa * sign
			MyCudaCopy(dev_alfa, alfa, sizeof(double), cudaMemcpyHostToDevice, 77);
			*alfa = fabs(*alfa); // back to postive alfa because we modified him on gpu.
			int indexValues = indexerMiss * (k + 1); // n* (K+1)
			int * dev_index_values = NULL;
			
			//index_values , the pointer for coordinates of point ,note values array size n*(k+1) 
			MyCudaMalloc((void**)&dev_index_values, sizeof(int), 88);
			MyCudaCopy(dev_index_values, &indexValues, sizeof(int), cudaMemcpyHostToDevice, 99);
			

			//create a new weight
			createNewWeight << <1, k + 1 >> > (dev_alfa, dev_values, dev_index_values, dev_W);

			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "createNewWeight launch failed: %s\n", cudaGetErrorString(cudaStatus));
				FreeFunction(dev_W, dev_alfa, dev_mislead);
				return -2;

			}
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching createNewWeight! : %s \n", cudaStatus, cudaGetErrorString(cudaStatus));
				FreeFunction(dev_W, dev_alfa, dev_mislead);
				return -2;

			}
		}
		counter_limit++;
	}

	//// need to calcate the q 
	
	int sumOFmisLead = 0;
	for (int i = 0; i < n; i++)
	{
		if (mislead[i] != 0)
			sumOFmisLead += 1;
	}

	double q = sumOFmisLead / (n*(1.0));
	MyCudaCopy(*WSaved, dev_W, sizeof(double)*(k + 1), cudaMemcpyDeviceToHost, 70); // copy W

	// clear resources on cuda GPU
	FreeFunction(dev_W, dev_alfa, dev_mislead);

	if (q <= QC)
		return q;
	else
		return 2.0; // q that never will get and larger from all q possiblies .



}
