

__global__
void MatrixMulKernel(float* M, float* N, float* P, int Width){
    // Calculate the row index of the P element and M
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the column index of P element and N
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < width && Col < Width) {
        float Pvalue = 0;

        // each thread computes one element of the block sub-matrix
        for (int k = 0; k < Width; ++k){
            Pvalue += M[Row*Width + k] * N[k*Width + Col];
        }

        P[Row*Width + Col] = Pvalue;
    }
}



// ° Host code to launch the kernel is shown below.

// • The configuration parameter dimGrid is set to ensure that for any combination of Width and
// BLOCK_WIDTH values, there are enough thread blocks in both x and y dimensions to calculate
// all P elements.

// this can be easily changed - 16 x 16 blocks
#define BLOCK_WIDTH 16
    // Set up the execution configuration
    int NumBlocks = WIDTH / BLOCK_WIDTH;
    if (WIDTH % BLOCK_WIDTH) NumBlocks++;

    dim3 dimGrid(NumBlocks, NumBlocks);
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);

    // Launch the device computation threads!
    MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);




// QUERYING DEVICE PROPERTIES
//Fields include 
dev_prop.maxThreadsPerBlock; 
dev_prop.multiProcessorCount; 
dev_prop.clockRate;
dev_prop.maxThreadsDim(0);   // for x dimension, to show the maximal number of threads along a dimension
dev_prop.maxGridSize(1);     // for y dimension, etc.; and others including dev_prop.warpSize
dev_prop.regsPerBlock; 
dev_prop.sharedMemPerBlock;
