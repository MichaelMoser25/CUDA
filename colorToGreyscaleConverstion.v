// We have 3 channels correspondign to RGB. The input image is encoded as unsigned characters [0, 255]
__global__
void colorToGreyscaleConverstion(insigned char *Pin, unisgned char *Pout, int width, int height){
    // Calkculate the row and column # of the Pin & Pout elements to process
    int Row = blockIdx.y*blockDim.y + threadIdx.y;
    int Col = blockIdx.x*blockDim.x + threadIdx.x;

    // Each thread computes one element of Pout if in range
    if (Row < height && Col < width) {  // only if threads within range
        int greyOffset = Row * width + Col; // get 1D coordinate for greyscale image
        // Think of the RGB image having CHANNEL times columns of the greyscale image
        int rgbOffset = greyOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];       // red value for pixel
        unsigned char g = Pin[rgbOffset + 1];   // green value for pixel
        unsigned char b = Pin[rgbOffset + 2];   // blue value for pixel

        // Perform the rescaling and store it. We multiply by floating-point constants
        Pout[grayOffset] = 0.21f*r + 0.72f*g + 0.07f*b;
    }
}
