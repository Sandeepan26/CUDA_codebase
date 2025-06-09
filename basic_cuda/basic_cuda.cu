//library includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//std input output 
#include "stdio.h"

/*
    This code below is a simple example of CUDA programming
    
    __global__: this qualifier is for specifying the function to be
    loaded onto the GPU
    
    basic: it is a simple function which prints the block and
    thread IDs during execution
    
    The numbers of blocks and threads are provided in the 
    main() function
*/
__global__ void basic()
{
    printf("Block Id is:  %d \t Thread ID is : %d \n", blockIdx.x, threadIdx.x);

    /*
        The block and thread IDs are not random
        In this case, they are specific in number based on the architecture

    */
}

int main()
{
    basic<<<2, 4>>>(); //number of blocks:=2, threads per block:=4

    /*
        cudaDeviceSynchronize() lets the CPU know about waiting
        for the GPU to finish the instructions and synchronize
        with the outputs 
    */
    cudaDeviceSynchronize();

    return 0;
}