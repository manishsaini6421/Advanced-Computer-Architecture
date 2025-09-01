#include "matrix_operation.h"
#include <immintrin.h>

Matrix MatrixOperation::NaiveMatMul(const Matrix &A, const Matrix &B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
	for(int i = 0; i < n ; i++) {
		for (int j = 0 ; j< m ; j++) {
			for(int l = 0; l < k; l++) {
				C(i,j) += A(i,l) * B(l,j);
			}
		}
	}
	
	return C;
}

// Loop reordered matrix multiplication (ikj order for better cache locality)
Matrix MatrixOperation::ReorderedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
	size_t k = A.getCols();
	size_t m = B.getCols();

	if (k != B.getRows()) {
		throw std::invalid_argument("Matrix dimensions don't match for multiplication");
	}
	
	
	Matrix C(n,m);
	
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (size_t i = 0; i < n; i++) {
        for (size_t l = 0; l < k; l++) {
            double r=A(i,l);
            for (size_t j = 0; j < m; j++) {
                C(i, j) += r * B(l, j);
            }
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------


	return C;
}

// Loop unrolled matrix multiplication
Matrix MatrixOperation::UnrolledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);

    const int UNROLL = 4;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            size_t l = 0;
            for (; l + UNROLL <= k; l += UNROLL) {
                C(i, j) += A(i, l) * B(l, j)
                         + A(i, l + 1) * B(l + 1, j)
                         + A(i, l + 2) * B(l + 2, j)
                         + A(i, l + 3) * B(l + 3, j);
            }
            for (; l < k; l++) {
                C(i, j) += A(i, l) * B(l, j);
            }
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Tiled (blocked) matrix multiplication for cache efficiency
Matrix MatrixOperation::TiledMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
    const int T = 16;   // tile size
	int i_max = 0;
	int k_max = 0;
	int j_max = 0;
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (size_t i = 0; i < n; i += T) {
        for (size_t j = 0; j < m; j += T) {
            for (size_t l = 0; l < k; l += T) {
                for (size_t ii = i; ii < i + T && ii < n; ++ii) {
                    for (size_t jj = j; jj < j + T && jj < m; ++jj) {
                        for (size_t ll = l; ll < l + T && ll < k; ++ll) {
                            C(ii, jj) += A(ii, ll) * B(ll, jj);
                        }
                    }
                }
            }
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// SIMD vectorized matrix multiplication (using AVX2)
Matrix MatrixOperation::VectorizedMatMul(const Matrix& A, const Matrix& B) {
	size_t n = A.getRows();
    size_t k = A.getCols();
    size_t m = B.getCols();

    if (k != B.getRows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }

    Matrix C(n, m);
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            __m256d c_vec = _mm256_setzero_pd();

            size_t l = 0;
            for (; l + 3 < k; l += 4) {
                __m256d a_vec = _mm256_set_pd(A(i, l + 3), A(i, l + 2),
                                              A(i, l + 1), A(i, l));
                __m256d b_vec = _mm256_set_pd(B(l + 3, j), B(l + 2, j),
                                              B(l + 1, j), B(l, j));
                c_vec = _mm256_add_pd(c_vec, _mm256_mul_pd(a_vec, b_vec));
            }

            double temp[4];
            _mm256_storeu_pd(temp, c_vec);
            C(i, j) += temp[0] + temp[1] + temp[2] + temp[3];

            for (; l < k; l++) {
                C(i, j) += A(i, l) * B(l, j);
            }
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------

    return C;
}

// Optimized matrix transpose
Matrix MatrixOperation::Transpose(const Matrix& A) {
	size_t rows = A.getRows();
	size_t cols = A.getCols();
	Matrix result(cols, rows);

	for (size_t i = 0; i < rows; ++i) {
		for (size_t j = 0; j < cols; ++j) {
			result(j, i) = A(i, j);
		}
	}

	// Optimized transpose using blocking for better cache performance
	// This is a simple implementation, more advanced techniques can be applied
	// Write your code here and commnent the above code
//----------------------------------------------------- Write your code here ----------------------------------------------------------------
	const size_t tile_size = 64; // Chosen to fit L1d cache (32 KB)
    for (size_t i = 0; i < rows; i += tile_size) {
        for (size_t j = 0; j < cols; j += tile_size) {
            for (size_t ii = i; ii < i + tile_size && ii < rows; ++ii) {
                for (size_t jj = j; jj < j + tile_size && jj < cols; ++jj) {
                    result(jj, ii) = A(ii, jj);
                }
            }
        }
    }

//-------------------------------------------------------------------------------------------------------------------------------------------

	
	return result;
}
