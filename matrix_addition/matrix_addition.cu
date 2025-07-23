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
        int idx = r * num_rows + c;
        result_matrix[idx] = input_matrix_a[idx] + input_matrix_b[idx];
    }
}

int main()
{
    return 0;
}