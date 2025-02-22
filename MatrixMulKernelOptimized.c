__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width)
{
    __shared__ float Mds[TITLE_WIDTH][TILE_WIDTH]; // Scope is block, into shared memory
    __shared__ float Nds[TITLE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;    int by = blockIdx.y;    // Scope is thread, into registers
    int tx = threadIdx.x;   int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty; // Identify the row index and column index
    int Col = bx * TILE_WIDTH + tx; // of the P element to work on
    float Pvalue = 0;

    // Loop over the M and N tiles required to compute the P element
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
        // Collaborative loading of M and N tiles into shared memory – See pages 24 and 25
        Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*Width + Col];

        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k) // Perform one phase of dot product
            Pvalue += Mds[ty][k] * Nds[k][tx];
        __synchthreads();
    }
    P[Row*Width + Col] = Pvalue; // All threads write to their P element
}
