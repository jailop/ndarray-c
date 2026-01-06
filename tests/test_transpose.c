/**
 * test_transpose.c - Tests for transpose
 */

#include "test_common.h"

void test_ndarray_transpose_2d(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new(dims);
    
    // A = [[1, 2, 3], [4, 5, 6]]
    A->data[0] = 1; A->data[1] = 2; A->data[2] = 3;
    A->data[3] = 4; A->data[4] = 5; A->data[5] = 6;
    
    NDArray B = ndarray_new_transpose(A);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 2);
    CU_ASSERT_EQUAL(B->dims[0], 3);
    CU_ASSERT_EQUAL(B->dims[1], 2);
    
    // B = [[1, 4], [2, 5], [3, 6]]
    CU_ASSERT_DOUBLE_EQUAL(B->data[0], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[1], 4.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[2], 2.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[3], 5.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[4], 3.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[5], 6.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_transpose_3d(void) {
    size_t dims[] = {2, 3, 4, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 24.0, 1.0);
    NDArray B = ndarray_new_transpose(A);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 3);
    CU_ASSERT_EQUAL(B->dims[0], 4);
    CU_ASSERT_EQUAL(B->dims[1], 3);
    CU_ASSERT_EQUAL(B->dims[2], 2);
    
    // Check a few elements
    size_t pos_a[] = {0, 0, 0};
    size_t pos_b[] = {0, 0, 0};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(B, pos_b), ndarray_get(A, pos_a), EPSILON);
    
    size_t pos_a2[] = {1, 2, 3};
    size_t pos_b2[] = {3, 2, 1};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(B, pos_b2), ndarray_get(A, pos_a2), EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_transpose_4d(void) {
    size_t dims[] = {2, 3, 4, 5, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 120.0, 1.0);
    NDArray B = ndarray_new_transpose(A);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 4);
    CU_ASSERT_EQUAL(B->dims[0], 5);
    CU_ASSERT_EQUAL(B->dims[1], 4);
    CU_ASSERT_EQUAL(B->dims[2], 3);
    CU_ASSERT_EQUAL(B->dims[3], 2);
    
    // Check corner elements
    size_t pos_a[] = {0, 0, 0, 0};
    size_t pos_b[] = {0, 0, 0, 0};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(B, pos_b), ndarray_get(A, pos_a), EPSILON);
    
    size_t pos_a2[] = {1, 2, 3, 4};
    size_t pos_b2[] = {4, 3, 2, 1};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(B, pos_b2), ndarray_get(A, pos_a2), EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
}

/* Register tests for this module */
void register_transpose_tests(CU_pSuite suite) {
    CU_add_test(suite, "test transpose 2d", test_ndarray_transpose_2d);
    CU_add_test(suite, "test transpose 3d", test_ndarray_transpose_3d);
    CU_add_test(suite, "test transpose 4d", test_ndarray_transpose_4d);
}
