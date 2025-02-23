° Assume we have already loaded array into global memory
__shared__ float partialSum[Size]
partialSum[threadIdx.x] = array[blockIdx.x*blockDim.x + threadIdx.x];// load the elements into shared memory

unsigned int t = threadIdx.x;
for (unsigned int stride = blockDim.x/2; stride >= 1; stride = stride >> 1) {
    __syncthreads(); // ensure all partial sums for the previous
    // iteration have been generated
    if (t < stride) // Good divergence
        partialSum[t] += partialSum[t + stride];
}
// the number of threads executing in each iteration is the same
// as before; however, the positions of threads that execute the
// addition relative to those that do not are different than the
// previous kernel.
