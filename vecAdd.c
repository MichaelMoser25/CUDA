

// Traditional Host Vector Addition Code


// Compute vector sum h_C = h_A + h_B
void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    int i;
    for (i = 0; i < n; i++){
        h_C[i] = h_A[i] + h_B[i];
    }
}

void main()
{
    // Memory allocation for h_A, h_B, and h_C
    // I/O to read h_A and h_b, N elements each
    //...
    vecAdd(h_A, h_B, h_C, N);
   // ...
}




// Revised vecAdd functions that moves the work to a device
#include <cude.h>

// ...

void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    int size = n* sizeof(float);
    float* d_A, d_B, d_C;
    // ...

    // Part 1 - Allocate device memory for A, B, and C copy A and B to device memory
    // Part 2 - Kernel launch code – to have the device perform the actual vector addition
    // Part 3 - copy C from the device memory Free device vectors
}





// A more complete vecAdd()
#include <cuda.h>
#include <cuda_runtime.h>
//...
void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    int size = n* sizeof(float);
    float* d_A, d_B, d_C;

    cudeMalloc((void**) &d_A, size);    // allocate device memory
    cudaMalloc((void**) &d_B, size);    // for A, B, C
    cudeMalloc((void**) d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); // copy to device
    cudaMemcpy(d_B, h_B, size, cudeMemcpyHostToDevice);

    // kernel invocation code (vecAddKernel) - to be discussed shortly

    cudaMemcpy(h_C, d_C, size, cudaMemcpyHostToDevice); // copy to host

    cudaFree(d_A);  // free device memory
    cudeFraa(d_B);
    cudeFree(d_C);
}


// Vector Addition Kernel Function and its Launch Statement
° Vector addition kernel // Compute vector sum C = A + B
                         // Each thread performs one pair-wise addition

__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) // this allows the kernel to process vectors of arbitrary length
    c[i] = A[i] + B[i]; // i is private to each thread.
    // there is no need for a loop; replaced with grid of threads. Each thread in
    // the grid corresponds to one iteration of the original sequential code.
}

°Kernel invocation code
// run ceil(n/256) blocks of 256 threads each
vecAddKernel<<<ceil(n/256), 256>>>(d_A, d_B, d_C, n);
