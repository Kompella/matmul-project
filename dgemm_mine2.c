// 10/01/2020: loop order is added
// 10/01/2020: blocking is added
// 10/02/2020: copy optimization is added
// Author: Jinwei Zhang (jz853@cornell.edu)

#include "dgemm_mine.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdalign.h>
#include <string.h>

const char* dgemm_desc = "My awesome dgemm2.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 64)  // 64 for n2 instance ?
#endif

void inner_multipy(const double* restrict Ab, const double* restrict Bb, double* restrict Cb)
{
    for (int j = 0; j < BLOCK_SIZE; ++j){
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            double Bj = Bb[j*BLOCK_SIZE+k];
            for (int i = 0; i < BLOCK_SIZE; ++i){
                // if (j < N && i < M) Cb[j*BLOCK_SIZE+i] += Ab[k*BLOCK_SIZE+i] * Bj;
                Cb[j*BLOCK_SIZE+i] += Ab[k*BLOCK_SIZE+i] * Bj;
            }
        }
    }
}

// jki order with block copy
void basic_dgemm(const int lda, const int M, const int N, const int K,
                 const double* restrict A, const double* restrict B, double* restrict C)
{   
    // copy A and B blocks to contiguous D 
    // double* D = (double*) malloc((M * K + N * K) * sizeof(double));
    alignas(16) double Ab[BLOCK_SIZE * BLOCK_SIZE];
    alignas(16) double Bb[BLOCK_SIZE * BLOCK_SIZE];
    alignas(16) double Cb[BLOCK_SIZE * BLOCK_SIZE];

    memset(Ab, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    memset(Bb, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));
    memset(Cb, 0, BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

    int a, b, c;
    for (c = 0; c < BLOCK_SIZE; ++c){
        for (a = 0; a < BLOCK_SIZE; ++a){
            if (c < K && a < M) Ab[c*BLOCK_SIZE+a] = A[c*lda+a];
            // Ab[c*BLOCK_SIZE+a] = A[c*lda+a];
        }
    }
    for (b = 0; b < BLOCK_SIZE; ++b){
        for (c = 0; c < BLOCK_SIZE; ++c){
            if (b < N && c < K) Bb[b*BLOCK_SIZE+c] = B[b*lda+c];
            // Bb[b*BLOCK_SIZE+c] = B[b*lda+c];
        }
    }
    for (b = 0; b < BLOCK_SIZE; ++b){
        for (a = 0; a < BLOCK_SIZE; ++a){
            if (b < N && a < M) Cb[b*BLOCK_SIZE+a] = C[b*lda+a];
            // Cb[b*BLOCK_SIZE+a] = C[b*lda+a];
        }
    }

    // do multiplication using contiguous Blocks
    inner_multipy(Ab, Bb, Cb);

    // copy block data back to C
    for (b = 0; b < BLOCK_SIZE; ++b){
        for (a = 0; a < BLOCK_SIZE; ++a){
            if (b < N && a < M) C[b*lda+a] = Cb[b*BLOCK_SIZE+a];
            // C[b*lda+a] = Cb[b*BLOCK_SIZE+a];
        }
    }

}

void do_block(const int lda,
              const double* restrict A, const double* restrict B, double* restrict C,
              const int i, const int j, const int k)
{
    const int M = (i+BLOCK_SIZE > lda? lda-i : BLOCK_SIZE);
    const int N = (j+BLOCK_SIZE > lda? lda-j : BLOCK_SIZE);
    const int K = (k+BLOCK_SIZE > lda? lda-k : BLOCK_SIZE);
    basic_dgemm(lda, M, N, K,
                A + i + k*lda, B + k + j*lda, C + i + j*lda);
}

void square_dgemm(const int M, const double* restrict A, const double* restrict B, double* restrict C)
{
    const int n_blocks = M / BLOCK_SIZE + (M%BLOCK_SIZE? 1 : 0);
    int bi, bj, bk;
    for (bj = 0; bj < n_blocks; ++bj) {
        const int j = bj * BLOCK_SIZE;
        for (bk = 0; bk < n_blocks; ++bk) {
            const int k = bk * BLOCK_SIZE;
            for (bi = 0; bi < n_blocks; ++bi) {
                const int i = bi * BLOCK_SIZE;
                do_block(M, A, B, C, i, j, k);
            }
        }
    }
}