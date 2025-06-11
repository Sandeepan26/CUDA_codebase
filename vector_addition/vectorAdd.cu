/*
    * This code is about Vector Addition in CUDA
    * the idea is to add elements of two vectors 
    * and store thee result in a resultant vector 
    * in the position corresponding to the elements 
    * taken as operands
    * C[i] = A[i] + B[i];

    * The data has to be first copied to the GPU and then the kernel to 
    * be called for the operation. Upon completion,
    * the data has to be copied from the GPU.
    * Steps:
    * Define macros, function for addition
    * Allocate memory for the vectors in the CPU as well as GPU
    * Initialize values for the vectors
    * Copy the values to the GPU
    * Call the kernel for the operation
    * Copy the result back to vector in the CPU
    
*/

//CUDA library imports
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//I/O imports
#include "stdio.h"

//preprocessor directives
#define num_blocks  1
#define num_threads_per_block 1024
#define vector_size 1024

/*--------------------START OF CODE------------------------ */
/*
    * __global__ is for CUDA to intepret that the function/kernel
    * is to be run on CUDA
    * void is to return nothing, but perform the operation
    * vectorAdd: function name which takes three arguments
    * vec1, vec2: input vector pointers
    * res: output vector where the result will be stored
    * the function will return an vectorAdd on the
    * number of threads it is operated on
*/
__global__ void vectorAdd(int *vec1, int *vec2, int *res, int size_vec)
{   
    if(threadIdx.x < vector_size)
    {
        res[threadIdx.x] = vec1[threadIdx.x] + vec2[threadIdx.x];
    }
}

//main function
int main()
{   
    int size = vector_size * sizeof(int);
    /*
        * Allocating memory to Host/CPU vectors
        * They are allocated memory as in C
        * pointer memory allocation
    */
    int *vector_1, *vector_2, *vector_3;

    vector_1 = (int *)malloc(size);
    vector_2 = (int *)malloc(size);
    vector_3 = (int *)malloc(size);

    /*
        * Allocation memory to device vectors
        * these vectors will be loaded onto the GPU
        * these are allocated memory by cudamalloc 
        * dynamic memory allocation
    */
    int *device_vec_1, *device_vec_2, *device_vec_3; //device vectors

    cudaMalloc((void **)&device_vec_1, size);
    cudaMalloc((void **)&device_vec_2, size);
    cudaMalloc((void **)&device_vec_3, size);

    //initializing vectors

    for(size_t i = 0; i < vector_size; ++i)
    {
        vector_1[i] = (i<<1); 
        vector_2[i] = ((i<<1) + 1); 
    }

    /*
        * Copying data to GPU, done via cudaMemcpy method
        * Syntax: cudaMemcpy(<destination>, <source>, <size>, <Copy_type);
        * Copy type here is HostToDevice 
    */

    cudaMemcpy(device_vec_1, vector_1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_vec_2, vector_2, size, cudaMemcpyHostToDevice);

    //calling function vectorAdd
    vectorAdd<<<num_blocks, num_threads_per_block>>>(device_vec_1, device_vec_2, device_vec_3, size);

    //storing the result of the vector addition
    cudaMemcpy(vector_3, device_vec_3, size, cudaMemcpyDeviceToHost);

    printf("Execution Successful\n");
    for(size_t iter = 0; iter < vector_size; ++iter)
    {
        printf("Input vectors A: %d \t B: %d \t Result: %d \n", vector_1[iter], vector_2[iter], vector_3[iter]);
    }
    //freeing host memory
    free(vector_1);
    free(vector_2);
    free(vector_3);

    //freeing device memory
    cudaFree(device_vec_1);
    cudaFree(device_vec_2);
    cudaFree(device_vec_3);

    //cudaDeviceSynchronize(); //waiting for GPU to finish operations
    return 0;
}

/*--------------------END OF CODE---------------------------*/