#ifndef MINE_H    /* This is an "include guard" */
#define MINE_H    /* prevents the file from being included twice. */
                     /* Including a header file twice causes all kinds */
                     /* of interesting problems.*/

/**
 * This is a function declaration.
 * It tells the compiler that the function exists somewhere.
 */
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double* A, const double* B, double* C);

#endif /* MINE_H */