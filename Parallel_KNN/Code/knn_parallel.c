// #include <stdio.h>
// #include <stdlib.h>
// #include <math.h>
// #include <mpi.h>
// #include <string.h>
// #include "helper.h"
// #include "mergeSort.h"

// // config file, make changes here for different dataset
// #include "config.h"

// void mpiInitialise(int *size, int *rank)
// {
// 	MPI_Init(NULL, NULL);
// 	MPI_Comm_rank(MPI_COMM_WORLD, rank);
// 	MPI_Comm_size(MPI_COMM_WORLD, size);
// }

// float *initFeatures(char path[])
// {
// 	int index = 0;
// 	FILE *f  = NULL;
// 	float *mat = NULL;

// 	mat = getFloatMat(NTRAIN, NFEATURES);

// 	f = fopen(path, "r");
// 	checkFile(f);

// 	while (fscanf(f, "%f%*c", &mat[index]) == 1) //%*c ignores the comma while reading the CSV
// 		index++;

// 	fclose(f);
// 	return mat;
// }

// float *initLabels(char path[])
// {
// 	int index = 0;
// 	FILE *f  = NULL;
// 	float *mat = NULL;

// 	mat = getFloatMat(NTRAIN, 1);

// 	f = fopen(path, "r");
// 	checkFile(f);

// 	while (fscanf(f, "%f%*c", &mat[index]) == 1)
// 		index++;

// 	fclose(f);
// 	return mat;
// }

// int predict(float *distance, float *labels) //topn < NCLASSES
// {
// 	float* neighborCount = getFloatMat(NCLASSES, 1);
// 	float* probability = getFloatMat(NCLASSES, 1);

// 	int i;

// 	for(i=0; i<K; i++)
// 		neighborCount[(int)labels[i]]++;

// 	for(i=0; i<NCLASSES; i++)
// 		probability[i] = neighborCount[i]*1.0/(float)K*1.0;
	
// 	int predicted_class = (int)getMax(neighborCount, NCLASSES);

// 	printf("Probability:\n");
// 	for(i=0; i<TOPN; i++)
// 		printf("%s\t%f\n", class[i], probability[i]);

// 	free(neighborCount);
// 	free(probability);

// 	return predicted_class;
// }

// void calcDistance(int ndata_per_process, float *pdistance, float *pdata, float *x)
// {
// 	int index = 0, i, j;
// 	for(i=0; i<ndata_per_process; i=i+NFEATURES)
// 	{
// 		pdistance[index] = 0.0;

// 		for(j=0; j<NFEATURES; j++)
// 			pdistance[index] = pdistance[index] + (pdata[i+j]-x[j])*(pdata[i+j]-x[j]);

// 		index++;
// 	}
// }

// void fit(float *X_train, float *y_train, float *X_test, float *y_test, int rank, int size)
// {
// 	int i, j;
// 	int ndata_per_process, nrows_per_process;
// 	float *pdata, *distance, *pdistance;
// 	float *plabels;
// 	float *labels;

// 	if (NTRAIN % size != 0)
// 	{
// 		if (rank == 0)
// 			printf("Number of rows in the training dataset should be divisibe by number of processors\n");
		
// 		MPI_Finalize();
// 		exit(0);
// 	}

// 	// initialise arrays
// 	nrows_per_process = NTRAIN/size;
// 	ndata_per_process = nrows_per_process*NFEATURES;

// 	pdata = getFloatMat(ndata_per_process, 1);
// 	pdistance = getFloatMat(nrows_per_process, 1);
// 	distance = getFloatMat(NTRAIN, 1);

// 	plabels = getFloatMat(nrows_per_process, 1);
// 	labels = getFloatMat(NTRAIN, 1);

// 	MPI_Scatter(X_train, ndata_per_process, MPI_FLOAT, pdata, ndata_per_process, MPI_FLOAT, 0,  MPI_COMM_WORLD);

// 	float *x = getFloatMat(NFEATURES, 1);

// 	for (i=0; i<NTEST; i=i+1)
// 	{	
// 		// very imp to scatter everytime in the loop here since plabels keep getting sorted and associativity is changed. 
// 		MPI_Scatter(y_train, nrows_per_process, MPI_FLOAT, plabels, nrows_per_process, MPI_FLOAT, 0,  MPI_COMM_WORLD);

// 		for(j=0; j<NFEATURES; j++)
// 			x[j] = X_test[i*NFEATURES+j];


// 		// fit
// 		calcDistance(ndata_per_process, pdistance, pdata, x);

// 		//sort the distance array 
// 		mergeSort(pdistance, 0, nrows_per_process - 1, plabels);

// 		MPI_Gather(pdistance, nrows_per_process, MPI_FLOAT, distance, nrows_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);
// 		MPI_Gather(plabels, nrows_per_process, MPI_FLOAT, labels, nrows_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);


// 		if (rank == 0)
// 		{
// 			mergeSort(distance, 0, NTRAIN - 1, labels);
// 			int predicted_class = predict(distance, labels);
// 			printf("%d) Predicted label: %d   True label: %d\n\n", i, predicted_class, (int)y_test[i]);
// 		}
// 	}

// 	free(x);
// 	free(distance);
// 	free(pdistance);
// }

// void knn(char *X_train_path, char *y_train_path, char *X_test_path, char *y_test_path)
// {
// 	float *X_train;
// 	float *y_train;
// 	float *X_test;
// 	float *y_test;
// 	double t1, t2;
// 	int size, rank;

// 	mpiInitialise(&size, &rank);
// 	//if(rank == 0) {
// 		if (rank == 0)
// 		{
// 			X_train = initFeatures(X_train_path);
// 			y_train = initLabels(y_train_path);
// 		}

// 		X_test = initFeatures(X_test_path);
// 		y_test = initLabels(y_test_path);

// 		if (rank == 0)
// 			t1 = MPI_Wtime();
		
		
// 		fit(X_train, y_train, X_test, y_test, rank, size);

// 		if (rank == 0)
// 			t2 = MPI_Wtime();

// 		if (rank == 0)
// 		{
// 			printf("Time for parallel execution (%d Processors): %f\n", size, t2 - t1);
// 			free(X_train);
// 			free(y_train);
// 		}
// 	//}
// 	free(X_test);
// 	free(y_test);
	
// 	MPI_Finalize();
// }

// int main()
// {
// 	knn(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH);
// 	return 0;
// }


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#include "helper_header.h"
#include "merge_sort_header.h"
#include "config_header.h"

/*
size -> req_size
rank -> req_rank

path -> path_to_file
*/





void initialise_mpi(int* req_size, int* req_rank) { // mpiInitialise
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, req_size);
	MPI_Comm_rank(MPI_COMM_WORLD, req_rank);
}
float *initialise_labels(char path_to_file[]) {
	float *matrix = NULL;
    FILE *file_ptr  = NULL;

    int index = 0;

	matrix = get_float_matrix(NTRAIN, 1);

	file_ptr = fopen(path_to_file, "r");
	check_file(file_ptr);

	while (fscanf(file_ptr, "%f%*c", &matrix[index]) == 1)
		index++;

	fclose(file_ptr);
	return matrix;
}
float *initialise_features(char path_to_file[]) {
	float *matrix = NULL;
    FILE *file_ptr  = NULL;

    int index = 0;

	matrix = get_float_matrix(NTRAIN, NFEATURES);

	file_ptr = fopen(path_to_file, "r");
	check_file(file_ptr);

	while (fscanf(file_ptr, "%f%*c", &matrix[index]) == 1)
		index++;

	fclose(file_ptr);
	return matrix;
}
int get_predicted_class(float *distance, float *labels) { //topn < NCLASSES
	
	float* prob = get_float_matrix(NCLASSES, 1);
    float* neighbor_count = get_float_matrix(NCLASSES, 1);

	int i = 0;
    while(i < K) {
        neighbor_count[(int)labels[i]]++;
        i++;
    }	
    i = 0;
	while(i < NCLASSES) {
		prob[i] = neighbor_count[i]*1.0/(float)K*1.0;
        i++;
    }
	
	int predicted_class = (int)get_max(neighbor_count, NCLASSES);

	printf("Probability:\n");
	
    i = 0;
    while(i<TOPN) {
		printf("%s\t%f\n", class[i], prob[i]);
        i++;
    }

	free(neighbor_count);
	free(prob);

	return predicted_class;
}
void calculate_distance(int ndata_per_process, float *p_distance, float *p_data, float *x) {
	int index = 0, i = 0, j;
	while(i<ndata_per_process)
	{
		p_distance[index] = 0.0;
        j = 0;
		while(j<NFEATURES) {
			p_distance[index] = p_distance[index] + (p_data[i+j]-x[j])*(p_data[i+j]-x[j]);
            j++;
        }
		index++;
        i=i+NFEATURES;
	}
}
void fit(float *x_train, float *y_train, float *x_test, float *y_test, int req_rank, int req_size) {
	int i, j;
	int ndata_per_process, nrows_per_process;
	float *p_data, *distance, *p_distance;
	float *p_labels;
	float *labels;

	if (NTRAIN % req_size != 0)
	{
		if (req_rank == 0)
			printf("Number of rows in the training dataset should be divisibe by number of processors\n");
		
		MPI_Finalize();
		exit(0);
	}

	// initialise arrays
	nrows_per_process = NTRAIN/req_size;
	ndata_per_process = nrows_per_process*NFEATURES;

	p_data = get_float_matrix(ndata_per_process, 1);
	p_distance = get_float_matrix(nrows_per_process, 1);
	distance = get_float_matrix(NTRAIN, 1);

	p_labels = get_float_matrix(nrows_per_process, 1);
	labels = get_float_matrix(NTRAIN, 1);

	MPI_Scatter(x_train, ndata_per_process, MPI_FLOAT, p_data, ndata_per_process, MPI_FLOAT, 0,  MPI_COMM_WORLD);

	float *x = get_float_matrix(NFEATURES, 1);

	for (i=0; i<NTEST; i=i+1)
	{	
		// very imp to scatter everytime in the loop here since p_labels keep getting sorted and associativity is changed. 
		MPI_Scatter(y_train, nrows_per_process, MPI_FLOAT, p_labels, nrows_per_process, MPI_FLOAT, 0,  MPI_COMM_WORLD);

		for(j=0; j<NFEATURES; j++)
			x[j] = x_test[i*NFEATURES+j];


		// fit
		calculate_distance(ndata_per_process, p_distance, p_data, x);

		//sort the distance array 
		mergeSort(p_distance, 0, nrows_per_process - 1, p_labels);

		MPI_Gather(p_distance, nrows_per_process, MPI_FLOAT, distance, nrows_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);
		MPI_Gather(p_labels, nrows_per_process, MPI_FLOAT, labels, nrows_per_process, MPI_FLOAT, 0, MPI_COMM_WORLD);


		if (req_rank == 0)
		{
			mergeSort(distance, 0, NTRAIN - 1, labels);
			int predicted_class = get_predicted_class(distance, labels);
			printf("%d) Predicted label: %d   True label: %d\n\n", i, predicted_class, (int)y_test[i]);
		}
	}

	free(x);
	free(distance);
	free(p_distance);
}
void knn(char *x_train_path, char *y_train_path, char *x_test_path, char *y_test_path)
{
	float *x_train;
	float *y_train;
	float *x_test;
	float *y_test;
	double t1, t2;
	int req_size, req_rank;

	initialise_mpi(&req_size, &req_rank);
    if (req_rank == 0)
    {
        x_train = initialise_features(x_train_path);
        y_train = initialise_labels(y_train_path);
    }

    x_test = initialise_features(x_test_path);
    y_test = initialise_labels(y_test_path);

    if (req_rank == 0)
        t1 = MPI_Wtime();
    
    
    fit(x_train, y_train, x_test, y_test, req_rank, req_size);

    if (req_rank == 0)
        t2 = MPI_Wtime();

    if (req_rank == 0)
    {
        printf("Time for parallel execution (%d Processors): %f\n", req_size, t2 - t1);
        free(x_train);
        free(y_train);
    }
	free(x_test);
	free(y_test);
	
	MPI_Finalize();
}
int main()
{
	knn(X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH, Y_TEST_PATH);
	return 0;
}
