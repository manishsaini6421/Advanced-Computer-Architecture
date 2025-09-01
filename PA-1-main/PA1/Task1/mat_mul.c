/*******************************************************************
 * Author: <Name1>, <Name2>
 * Date: <Date>
 * File: mat_mul.c
 * Description: This file contains implementations of matrix multiplication
 *			    algorithms using various optimization techniques.
 *******************************************************************/

// PA 1: Matrix Multiplication

// includes
#include <stdio.h>
#include <stdlib.h>         // for malloc, free, atoi
#include <time.h>           // for time()
#include <chrono>	        // for timing
#include <xmmintrin.h> 		// for SSE
#include <immintrin.h>		// for AVX

#include "helper.h"			// for helper functions

// defines
// NOTE: you can change this value as per your requirement
#define TILE_SIZE	16 // size of the tile for blocking

/**
 * @brief 		Performs matrix multiplication of two matrices.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void naive_mat_mul(double *A, double *B, double *C, int size) {

	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			for (int k = 0; k < size; k++) {
				C[i * size + j] += A[i * size + k] * B[k * size + j];
			}
		}
	}
}

/**
 * @brief 		Task 1A: Performs matrix multiplication of two matrices using loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 */
void loop_opt_mat_mul(double *A, double *B, double *C, int size){
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
	//Effects of loops (kij)
	// for (int k = 0; k < size; k++) {
	// 	for (int i = 0; i < size; i++) {
	// 		double r=A[i * size + k];
	// 		for (int j = 0; j < size; j++) {
	// 			C[i * size + j] += r * B[k * size + j];
	// 		}
	// 	}
	// }

	// //Effects of loops (jki)
	// for (int j = 0; j < size; j++) {
	// 	for (int k = 0; k < size; k++) {
	// 		double r=B[k * size + j];
	// 		for (int i = 0; i < size; i++) {
	// 			C[i * size + j] += A[i * size + k] * r;
	// 		}
	// 	}
	// }


	// //Effects of loops (ikj)
	for (size_t i = 0; i < size; i++) {
        for (size_t k = 0; k < size; k++) {
            double r = A[i * size + k];
            for (size_t j = 0; j < size; j++) {
                C[i * size + j] += r * B[k * size + j];
            }
        }
    }

	//Effects of loop unrolling

	// for (int i = 0; i < size; i++) {
	// 	for (int j = 0; j < size; j++) {
	// 		double r = A[i * size + j];
	// 		for (int k = 0; k < size; k += 4) {
	// 			C[i * size + k] += r * B[j * size + k];
	// 			C[i * size + k + 1] += r * B[j * size + k + 1];
	// 			C[i * size + k + 2] += r * B[j * size + k + 2];
	// 			C[i * size + k + 3] += r * B[j * size + k + 3];
	// 		}
	// 	}
	// }

//-------------------------------------------------------------------------------------------------------------------------------------------

}


/**
 * @brief 		Task 1B: Performs matrix multiplication of two matrices using tiling.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * 				For example, if the dimension is 1024, then the tile size can be 32, 64, 128, etc.
 * 				You can assume that the matrices are square matrices.
*/
void tile_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    //tile_matrix_multiplication
   for (int i = 0; i < size; i += tile_size) {
        for (int j = 0; j < size; j += tile_size) {
            for (int k = 0; k < size; k += tile_size) {
                for (int ii = i; ii < i + tile_size && ii < size; ++ii) {
                    for (int jj = j; jj < j + tile_size && jj < size; ++jj) {
                        for (int kk = k; kk < k + tile_size && kk < size; ++kk) {
                            C[ii * size + jj] += A[ii * size + kk] * B[kk * size + jj];
                        }
                    }
                }
            }
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

/**
 * @brief 		Task 1C: Performs matrix multiplication of two matrices using SIMD instructions.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @note 		You can assume that the matrices are square matrices.
*/
void simd_mat_mul(double *A, double *B, double *C, int size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    // 128-bit SSE version (processes 2 doubles at a time)
	//  for (int i = 0; i < size; i++) {
    //     for (int j = 0; j < size; j++) {
    //         __m128d c_vec = _mm_setzero_pd(); // Initialize vector for C[i][j]
    //         for (int k = 0; k < size - 1; k += 2) { // Process 2 elements at a time
    //             __m128d a_vec = _mm_loadu_pd(&A[i * size + k]); // Load 2 elements from A[i][k:k+1]
    //             // Load B[k][j] and B[k+1][j] (non-contiguous) into a vector
    //             __m128d b_vec = _mm_set_pd(B[(k + 1) * size + j], B[k * size + j]);
    //             c_vec = _mm_add_pd(c_vec, _mm_mul_pd(a_vec, b_vec)); // c_vec = c_vec + (a_vec * b_vec)
    //         }
    //         // Reduce c_vec to scalar and store in C[i][j]
    //         double temp[2];
    //         _mm_storeu_pd(temp, c_vec);
    //         C[i * size + j] += temp[0] + temp[1];
    //         // Handle remainder if size is odd
    //         if (size % 2) {
    //             C[i * size + j] += A[i * size + (size - 1)] * B[(size - 1) * size + j];
    //         }
    //     }
    // }



	//256-bit AVX version (processes 4 doubles at a time)
	for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            __m256d c_vec = _mm256_setzero_pd(); // Initialize vector for C[i][j]
            for (int k = 0; k < size - 3; k += 4) { // Process 4 elements at a time
                __m256d a_vec = _mm256_loadu_pd(&A[i * size + k]); // Load 4 elements from A[i][k:k+3]
                // Load B[k][j], B[k+1][j], B[k+2][j], B[k+3][j] (non-contiguous) into a vector
                __m256d b_vec = _mm256_set_pd(B[(k + 3) * size + j], B[(k + 2) * size + j],
                                              B[(k + 1) * size + j], B[k * size + j]);
                c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_vec, b_vec)); // c_vec = c_vec + (a_vec * b_vec)
            }
            // Reduce c_vec to scalar and store in C[i][j]
            double temp[4];
            _mm256_storeu_pd(temp, c_vec);
            C[i * size + j] += temp[0] + temp[1] + temp[2] + temp[3];
            // Handle remainder if size % 4 != 0
            for (int k = size - (size % 4); k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

/**
 * @brief 		Task 1D: Performs matrix multiplication of two matrices using combination of tiling/SIMD/loop optimization.
 * @param 		A 			pointer to the first matrix
 * @param 		B 			pointer to the second matrix
 * @param 		C 			pointer to the resultant matrix
 * @param 		size 		dimension of the matrices
 * @param 		tile_size 	size of the tile
 * @note 		The tile size should be a multiple of the dimension of the matrices.
 * @note 		You can assume that the matrices are square matrices.
*/
void combination_mat_mul(double *A, double *B, double *C, int size, int tile_size) {
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    //SIMD + Tiling
	// for (size_t i = 0; i < size; i += tile_size) {
    //     for (size_t j = 0; j < size; j += tile_size) {
    //         for (size_t k = 0; k < size; k += tile_size) {
    //             for (size_t ii = i; ii < i + tile_size && ii < size; ++ii) {
    //                 for (size_t jj = j; jj < j + tile_size && jj < size; ++jj) {
    //                     __m256d c_vec = _mm256_setzero_pd(); // Initialize vector for C[ii][jj]
    //                     for (size_t kk = k; kk < k + tile_size - 3 && kk < size; kk += 4) { // Process 4 elements at a time
    //                         __m256d a_vec = _mm256_loadu_pd(&A[ii * size + kk]); // Load 4 elements from A[ii][kk:kk+3]
    //                         // Load B[kk][jj], B[kk+1][jj], B[kk+2][jj], B[kk+3][jj] (non-contiguous) into a vector
    //                         __m256d b_vec = _mm256_set_pd(B[(kk + 3) * size + jj], B[(kk + 2) * size + jj],
    //                                                       B[(kk + 1) * size + jj], B[kk * size + jj]);
    //                         c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_vec, b_vec)); // c_vec = c_vec + (a_vec * b_vec)
    //                     }
    //                     // Reduce c_vec to scalar and store in C[ii][jj]
    //                     double temp[4];
    //                     _mm256_storeu_pd(temp, c_vec);
    //                     C[ii * size + jj] += temp[0] + temp[1] + temp[2] + temp[3];
    //                     // Handle remainder if tile_size % 4 != 0 or kk < size
    //                     for (size_t kk = k + (tile_size - (tile_size % 4)); kk < k + tile_size && kk < size; kk++) {
    //                         C[ii * size + jj] += A[ii * size + kk] * B[kk * size + jj];
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
    

	//******SIMD + Tiling + Transpose*******
	// Allocate memory for B_transpose
    double* B_transpose = (double*)malloc(size * size * sizeof(double));
    if (!B_transpose) {
        // Handle allocation failure (in practice, add proper error handling)
        return;
    }

    // Transpose B
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            B_transpose[j * size + i] = B[i * size + j];
        }
    }

    // Tiling + SIMD matrix multiplication
    for (size_t i = 0; i < size; i += tile_size) {
        for (size_t j = 0; j < size; j += tile_size) {
            for (size_t k = 0; k < size; k += tile_size) {
                for (size_t ii = i; ii < i + tile_size && ii < size; ++ii) {
                    for (size_t jj = j; jj < j + tile_size && jj < size; ++jj) {
                        __m256d c_vec = _mm256_setzero_pd(); // Initialize vector for C[ii][jj]
                        for (size_t kk = k; kk < k + tile_size - 3 && kk < size; kk += 4) { // Process 4 elements
                            __m256d a_vec = _mm256_loadu_pd(&A[ii * size + kk]); // Load A[ii][kk:kk+3]
                            __m256d b_vec = _mm256_loadu_pd(&B_transpose[jj * size + kk]); // Load B^T[jj][kk:kk+3]
                            c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_vec, b_vec)); // c_vec += a_vec * b_vec
                        }
                        // Reduce c_vec to scalar and store in C[ii][jj]
                        double temp[4];
                        _mm256_storeu_pd(temp, c_vec);
                        C[ii * size + jj] += temp[0] + temp[1] + temp[2] + temp[3];
                        // Handle remainder
                        for (size_t kk = k + (tile_size - (tile_size % 4)); kk < k + tile_size && kk < size; kk++) {
                            C[ii * size + jj] += A[ii * size + kk] * B_transpose[jj * size + kk];
                        }
                    }
                }
            }
        }
    }

    // Free B_transpose
    free(B_transpose);
//-------------------------------------------------------------------------------------------------------------------------------------------
    
}

// NOTE: DO NOT CHANGE ANYTHING BELOW THIS LINE
/**
 * @brief 		Main function
 * @param 		argc 		number of command line arguments
 * @param 		argv 		array of command line arguments
 * @return 		0 on success
 * @note 		DO NOT CHANGE THIS FUNCTION
 * 				DO NOT ADD OR REMOVE ANY COMMAND LINE ARGUMENTS
*/
int main(int argc, char **argv) {

	if ( argc <= 1 ) {
		printf("Usage: %s <matrix_dimension>\n", argv[0]);
		return 0;
	}

	else {
		int size = atoi(argv[1]);

		double *A = (double *)malloc(size * size * sizeof(double));
		double *B = (double *)malloc(size * size * sizeof(double));
		double *C = (double *)calloc(size * size, sizeof(double));

		// initialize random seed
		srand(time(NULL));

		// initialize matrices A and B with random values
		initialize_matrix(A, size, size);
		initialize_matrix(B, size, size);

		// perform normal matrix multiplication
		
		//initialize result matrix to 0
		// initialize_result_matrix(C, size, size);

		// auto start = std::chrono::high_resolution_clock::now();
		// naive_mat_mul(A, B, C, size);
		// auto end = std::chrono::high_resolution_clock::now();
		// auto time_naive_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		// printf("Normal matrix multiplication took %ld ms to execute \n\n", time_naive_mat_mul);

	#ifdef OPTIMIZE_LOOP_OPT
		// Task 1a: perform matrix multiplication with loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		auto start = std::chrono::high_resolution_clock::now();
		loop_opt_mat_mul(A, B, C, size);
		auto end = std::chrono::high_resolution_clock::now();
		auto time_loop_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Loop optimized matrix multiplication took %ld ms to execute \n", time_loop_mat_mul);
		// printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_loop_mat_mul);
	#endif

	#ifdef OPTIMIZE_TILING
		// Task 1b: perform matrix multiplication with tiling

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		auto start = std::chrono::high_resolution_clock::now();
		tile_mat_mul(A, B, C, size, TILE_SIZE);
		auto end = std::chrono::high_resolution_clock::now();
		auto time_tiling_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Tiling matrix multiplication took %ld ms to execute \n", time_tiling_mat_mul);
		//printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_tiling_mat_mul);
	#endif

	#ifdef OPTIMIZE_SIMD
		// Task 1c: perform matrix multiplication with SIMD instructions 

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		auto start = std::chrono::high_resolution_clock::now();
		simd_mat_mul(A, B, C, size);
		auto end = std::chrono::high_resolution_clock::now();
		auto time_simd_mat_mul = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

		printf("SIMD matrix multiplication took %ld ms to execute \n", time_simd_mat_mul);
		// printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_simd_mat_mul);
	#endif

	#ifdef OPTIMIZE_COMBINED
		// Task 1d: perform matrix multiplication with combination of tiling, SIMD and loop optimization

		// initialize result matrix to 0
		initialize_result_matrix(C, size, size);

		auto start = std::chrono::high_resolution_clock::now();
		combination_mat_mul(A, B, C, size, TILE_SIZE);
		auto end = std::chrono::high_resolution_clock::now();
		auto time_combination = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		printf("Combined optimization matrix multiplication took %ld ms to execute \n", time_combination);
		// printf("Normalized performance: %f \n\n", (double)time_naive_mat_mul / time_combination);
	#endif

		// free allocated memory
		free(A);
		free(B);
		free(C);

		return 0;
	}
}
