/* widthA = heightB for valid matrix multiplication */
__kernel void simpleMultiply(
    __global float *outputD,
    int widthA,
    int heightA,
    int widthB,
    int heightB,
    int widthC,
    int heightC,
    __global float *inputA,
    __global float *inputB,
    __global float *inputC)
{
    /* Get global position in Y direction */
    int row = get_global_id (1);
    /* Get global position in X direction */
    int col = get_global_id (0);

    float sum = 0.0f;

    /* Calculate result of one element of Matrix A * Matrix B */
    for (int i=0; i < widthA; i++) {
        sum += inputA[row * widthA + i] * inputB[i * widthB + col];
    }
    /* Add results from Matrix A * Matrix B to Matrix C */
    outputD[row * widthB + col] = sum + inputC[row * widthB + col];
}
