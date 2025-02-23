// ° Assume we have already loaded the array into global memory
__shared__ float partialSum[Size]
partialSum[threadIdx.x] = array[blockIdx.x*blockDim.x + threadIdx.x]; // load the elements into shared memory
unsigned int t = threadIdx.x;
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
    __syncthreads(); // ensure all partial sums for the previous
    // iteration have been generated
    if (t % (2*stride) == 0) // Bad - divergence
        partialSum[t] += partialSum[t + stride];
}
// During the first iteration, only even threads perform addition
// between two neighbouring elements. During the 2nd iteration, only
// those threads whose indices are multiples of four will execute the
// add statement, etc.
