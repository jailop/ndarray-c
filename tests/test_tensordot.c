/**
 * test_tensordot.c - Tests for tensordot
 */

#include "test_common.h"

void test_ndarray_tensordot_2d_matmul(void) {
    // Test tensordot equivalent to matrix multiplication
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
    
    // Contract on last axis of A (axis 1) and first axis of B (axis 0)
    NDArray C = ndarray_new_tensordot(A, B, NDA_AXES(1), NDA_AXES(0));
    
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

void test_ndarray_tensordot_3d_single_axis(void) {
    // A: (2, 3, 4), B: (4, 5) -> C: (2, 3, 5)
    size_t dims_a[] = {2, 3, 4, 0};
    size_t dims_b[] = {4, 5, 0};
    
    NDArray A = ndarray_new_ones(dims_a);
    NDArray B = ndarray_new_full(dims_b, 2.0);
    
    // Contract on last axis of A (axis 2) and first axis of B (axis 0)
    NDArray C = ndarray_new_tensordot(A, B, NDA_AXES(2), NDA_AXES(0));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 3);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 3);
    CU_ASSERT_EQUAL(C->dims[2], 5);
    
    // All elements should be 8.0 (sum of 4 * 1.0 * 2.0)
    for (size_t i = 0; i < 30; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 8.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_tensordot_multi_axis(void) {
    // A: (2, 3, 4, 5), B: (4, 5, 6) -> contract axes (2,3) with (0,1) -> C: (2, 3, 6)
    size_t dims_a[] = {2, 3, 4, 5, 0};
    size_t dims_b[] = {4, 5, 6, 0};
    
    NDArray A = ndarray_new_ones(dims_a);
    NDArray B = ndarray_new_ones(dims_b);
    
    // Contract on axes 2,3 of A and axes 0,1 of B
    NDArray C = ndarray_new_tensordot(A, B, NDA_AXES(2, 3), NDA_AXES(0, 1));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 3);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 3);
    CU_ASSERT_EQUAL(C->dims[2], 6);
    
    // All elements should be 20.0 (4*5 contractions of 1.0*1.0)
    for (size_t i = 0; i < 36; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 20.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_tensordot_vs_matmul(void) {
    // Verify tensordot and matmul produce same results for matrix multiplication
    size_t dims_a[] = {3, 4, 0};
    size_t dims_b[] = {4, 5, 0};
    
    NDArray A = ndarray_new(dims_a);
    NDArray B = ndarray_new(dims_b);
    
    // Fill with sequential values
    for (size_t i = 0; i < 12; ++i) A->data[i] = i + 1.0;
    for (size_t i = 0; i < 20; ++i) B->data[i] = (i + 1.0) * 0.5;
    
    // Compute using matmul
    NDArray C_matmul = ndarray_new_matmul(A, B);
    
    // Compute using tensordot
    NDArray C_tensordot = ndarray_new_tensordot(A, B, NDA_AXES(1), NDA_AXES(0));
    
    CU_ASSERT_PTR_NOT_NULL(C_matmul);
    CU_ASSERT_PTR_NOT_NULL(C_tensordot);
    CU_ASSERT_EQUAL(C_matmul->ndim, C_tensordot->ndim);
    CU_ASSERT_EQUAL(C_matmul->dims[0], C_tensordot->dims[0]);
    CU_ASSERT_EQUAL(C_matmul->dims[1], C_tensordot->dims[1]);
    
    // Compare all elements
    for (size_t i = 0; i < 15; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C_matmul->data[i], C_tensordot->data[i], EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C_matmul);
    ndarray_free(C_tensordot);
}

void test_ndarray_tensordot_outer_product(void) {
    // Test outer product: contract 0 axes -> outer product
    size_t dims_a[] = {2, 3, 0};
    size_t dims_b[] = {3, 4, 0};
    
    NDArray A = ndarray_new_full(dims_a, 2.0);
    NDArray B = ndarray_new_full(dims_b, 3.0);
    
    // No axes contracted -> outer product
    NDArray C = ndarray_new_tensordot(A, B, NDA_NO_AXES, NDA_NO_AXES);
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 4);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 3);
    CU_ASSERT_EQUAL(C->dims[2], 3);
    CU_ASSERT_EQUAL(C->dims[3], 4);
    
    // All elements should be 6.0 (2.0 * 3.0)
    for (size_t i = 0; i < 72; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 6.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

/* Register tests for this module */
void register_tensordot_tests(CU_pSuite suite) {
    CU_add_test(suite, "test tensordot 2d matmul", test_ndarray_tensordot_2d_matmul);
    CU_add_test(suite, "test tensordot 3d single axis", test_ndarray_tensordot_3d_single_axis);
    CU_add_test(suite, "test tensordot multi axis", test_ndarray_tensordot_multi_axis);
    CU_add_test(suite, "test tensordot vs matmul", test_ndarray_tensordot_vs_matmul);
    CU_add_test(suite, "test tensordot outer product", test_ndarray_tensordot_outer_product);
}
