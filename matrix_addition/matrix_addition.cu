//CUDA Imports
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//macros
#define num_rows 4096
#define num_cols 4096

/*
    The code block below represents the kernel for matrix addition
    A 2-D matrix is represented by blocks spread across x and y dimensions
    This is a row-major approach, matrix addition starts with considering all columns 
    of a row first. Then the rows gradually move in y dimension and proceed with the 
    same operation
*/

__global__ void mat_add (float* input_matrix_a, float* input_matrix_b, float* result_matrix, int rows, int cols)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if((r < num_rows) && (c < num_cols))
    {   
        int idx = r * num_cols + c;
        result_matrix[idx] = input_matrix_a[idx] + input_matrix_b[idx];
    }
}

int main()
{   
    //allocating memory for matrices in the host device
    float *host_matrix_a = (float*)malloc(sizeof(float) * num_rows * num_cols);
    float *host_matrix_b = (float*)malloc(sizeof(float) * num_rows * num_cols);
    float *host_matrix_c = (float*)malloc(sizeof(float)* num_rows * num_cols);

    //allocating memory for memory in the gpu
    float *d_matrix_a, *d_matrix_b, *d_matrix_c;
    
    cudaMalloc((void**)&d_matrix_a, sizeof(float) * num_rows * num_cols);
    cudaMalloc((void**)&d_matrix_b, sizeof(float) * num_rows * num_cols);
    cudaMalloc((void**)&d_matrix_c, sizeof(float) * num_rows * num_cols);

    //Assigning values to the matrices
    for(int iter = 0; iter < num_rows * num_cols ; ++iter)
    {
        host_matrix_a[iter] = static_cast<float>(rand())/ static_cast<float>(RAND_MAX);
        host_matrix_b[iter] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }

    //defining the dimensions of the matrix
    dim3 blockDim(32, 32);
    dim3 gridDim((num_rows + blockDim.x - 1)/blockDim.x,(num_cols + blockDim.y - 1)/blockDim.y);

    //copying memory of host vectors to device vectors
    cudaMemcpy(d_matrix_a, host_matrix_a, num_rows * num_cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix_b, host_matrix_b, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice);

    //calling the kernel
    mat_add<<<gridDim, blockDim>>>(d_matrix_a, d_matrix_b, d_matrix_c, num_rows, num_cols);


    //copy the result from gpu
    cudaMemcpy(host_matrix_c, d_matrix_c, num_rows* num_cols * sizeof(float), cudaMemcpyDeviceToHost);

    //freeing memory
    free(host_matrix_a);
    free(host_matrix_b);
    free(host_matrix_c);

    cudaFree(d_matrix_a);
    cudaFree(d_matrix_b);
    cudaFree(d_matrix_c);
    
    cudaDeviceSynchronize();
    return 0;
}