/**
 * test_matmul.c - Tests for matrix multiplication
 */

#include "test_common.h"

void test_ndarray_matmul_2d(void) {
    // A: 2x3, B: 3x2 -> C: 2x2
    size_t dims_a[] = {2, 3, 0};
    size_t dims_b[] = {3, 2, 0};
    
    NDArray A = ndarray_new(dims_a);
    NDArray B = ndarray_new(dims_b);
    
    // A = [[1, 2, 3], [4, 5, 6]]
    A->data[0] = 1; A->data[1] = 2; A->data[2] = 3;
    A->data[3] = 4; A->data[4] = 5; A->data[5] = 6;
    
    // B = [[1, 2], [3, 4], [5, 6]]
    B->data[0] = 1; B->data[1] = 2;
    B->data[2] = 3; B->data[3] = 4;
    B->data[4] = 5; B->data[5] = 6;
    
    NDArray C = ndarray_new_matmul(A, B);
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 2);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 2);
    
    // C = [[22, 28], [49, 64]]
    CU_ASSERT_DOUBLE_EQUAL(C->data[0], 22.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[1], 28.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[2], 49.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[3], 64.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_matmul_3d_batch(void) {
    // Batch of 2 matrices: [2, 2, 3] @ [2, 3, 2] -> [2, 2, 2]
    size_t dims_a[] = {2, 2, 3, 0};
    size_t dims_b[] = {2, 3, 2, 0};
    
    NDArray A = ndarray_new_ones(dims_a);
    NDArray B = ndarray_new_ones(dims_b);
    
    NDArray C = ndarray_new_matmul(A, B);
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 3);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 2);
    CU_ASSERT_EQUAL(C->dims[2], 2);
    
    // All elements should be 3.0 (sum of 3 ones)
    for (size_t i = 0; i < 8; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 3.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_matmul_4d_batch(void) {
    // Batch: [2, 3, 2, 2] @ [2, 3, 2, 2] -> [2, 3, 2, 2]
    size_t dims[] = {2, 3, 2, 2, 0};
    
    NDArray A = ndarray_new_full(dims, 2.0);
    NDArray B = ndarray_new_full(dims, 0.5);
    
    NDArray C = ndarray_new_matmul(A, B);
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 4);
    
    // Each 2x2 result should have all 2.0s (2*0.5 + 2*0.5 = 2.0)
    for (size_t i = 0; i < 24; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 2.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

/* Register tests for this module */
void register_matmul_tests(CU_pSuite suite) {
    CU_add_test(suite, "test matmul 2d", test_ndarray_matmul_2d);
    CU_add_test(suite, "test matmul 3d batch", test_ndarray_matmul_3d_batch);
    CU_add_test(suite, "test matmul 4d batch", test_ndarray_matmul_4d_batch);
}
