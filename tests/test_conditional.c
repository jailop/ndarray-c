/**
 * test_conditional.c - Tests for conditional operations
 */

#include "test_common.h"

void test_ndarray_clip_min_basic(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_arange(dims, -3.0, 3.0, 1.0);
    
    // Clip to minimum 0.0
    ndarray_clip_min(A, 0.0);
    
    CU_ASSERT_DOUBLE_EQUAL(A->data[0], 0.0, EPSILON);  // was -3
    CU_ASSERT_DOUBLE_EQUAL(A->data[1], 0.0, EPSILON);  // was -2
    CU_ASSERT_DOUBLE_EQUAL(A->data[2], 0.0, EPSILON);  // was -1
    CU_ASSERT_DOUBLE_EQUAL(A->data[3], 0.0, EPSILON);  // was 0
    CU_ASSERT_DOUBLE_EQUAL(A->data[4], 1.0, EPSILON);  // was 1
    CU_ASSERT_DOUBLE_EQUAL(A->data[5], 2.0, EPSILON);  // was 2
    
    ndarray_free(A);
}

void test_ndarray_clip_max_basic(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_arange(dims, -1.0, 5.0, 1.0);
    
    // Clip to maximum 2.0
    ndarray_clip_max(A, 2.0);
    
    CU_ASSERT_DOUBLE_EQUAL(A->data[0], -1.0, EPSILON);  // was -1
    CU_ASSERT_DOUBLE_EQUAL(A->data[1], 0.0, EPSILON);   // was 0
    CU_ASSERT_DOUBLE_EQUAL(A->data[2], 1.0, EPSILON);   // was 1
    CU_ASSERT_DOUBLE_EQUAL(A->data[3], 2.0, EPSILON);   // was 2
    CU_ASSERT_DOUBLE_EQUAL(A->data[4], 2.0, EPSILON);   // was 3
    CU_ASSERT_DOUBLE_EQUAL(A->data[5], 2.0, EPSILON);   // was 4
    
    ndarray_free(A);
}

void test_ndarray_clip_range(void) {
    size_t dims[] = {2, 4, 0};
    NDArray A = ndarray_new_arange(dims, -2.0, 6.0, 1.0);
    
    // Clip to [-1, 3] range
    ndarray_clip(A, -1.0, 3.0);
    
    CU_ASSERT_DOUBLE_EQUAL(A->data[0], -1.0, EPSILON);  // was -2
    CU_ASSERT_DOUBLE_EQUAL(A->data[1], -1.0, EPSILON);  // was -1
    CU_ASSERT_DOUBLE_EQUAL(A->data[2], 0.0, EPSILON);   // was 0
    CU_ASSERT_DOUBLE_EQUAL(A->data[3], 1.0, EPSILON);   // was 1
    CU_ASSERT_DOUBLE_EQUAL(A->data[4], 2.0, EPSILON);   // was 2
    CU_ASSERT_DOUBLE_EQUAL(A->data[5], 3.0, EPSILON);   // was 3
    CU_ASSERT_DOUBLE_EQUAL(A->data[6], 3.0, EPSILON);   // was 4
    CU_ASSERT_DOUBLE_EQUAL(A->data[7], 3.0, EPSILON);   // was 5
    
    ndarray_free(A);
}

void test_ndarray_abs_basic(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_arange(dims, -2.0, 4.0, 1.0);
    
    ndarray_abs(A);
    
    CU_ASSERT_DOUBLE_EQUAL(A->data[0], 2.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[1], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[2], 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[3], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[4], 2.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[5], 3.0, EPSILON);
    
    ndarray_free(A);
}

void test_ndarray_sign_basic(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new(dims);
    A->data[0] = -5.0;
    A->data[1] = -0.5;
    A->data[2] = 0.0;
    A->data[3] = 0.5;
    A->data[4] = 5.0;
    A->data[5] = 100.0;
    
    ndarray_sign(A);
    
    CU_ASSERT_DOUBLE_EQUAL(A->data[0], -1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[1], -1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[2], 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[3], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[4], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[5], 1.0, EPSILON);
    
    ndarray_free(A);
}

/* Register tests for this module */
void register_conditional_tests(CU_pSuite suite) {
    CU_add_test(suite, "test clip min basic", test_ndarray_clip_min_basic);
    CU_add_test(suite, "test clip max basic", test_ndarray_clip_max_basic);
    CU_add_test(suite, "test clip range", test_ndarray_clip_range);
    CU_add_test(suite, "test abs basic", test_ndarray_abs_basic);
    CU_add_test(suite, "test sign basic", test_ndarray_sign_basic);
}
