const char* dgemm_desc = "My awesome dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)
#endif

#define min(a,b) (((a)<(b))?(a):(b))

// Algorithm from : https://www.cs.utexas.edu/users/flame/pubs/blis3_ipdps14.pdf

void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    const int n_blocks_size = n_blocks*BLOCK_SIZE;
    int jc, pc, jr, ir, ic; double temp;

    for (jc = 0; jc < n_blocks_size; jc += BLOCK_SIZE) {

        for (pc = 0; pc < n_blocks_size; pc += BLOCK_SIZE) {

            for (ic = pc; ic < min(M, pc + BLOCK_SIZE); ic++) {

		for (jr = jc; jr < min(M, jc + BLOCK_SIZE); jr++) {

		    temp = B[ic*M + jr];

		    for (ir = 0; ir < M; ir++){

			C[ic*M + ir] += A[jr*M + ir]*temp;

		    }
		}
            }
        }
    }
}

