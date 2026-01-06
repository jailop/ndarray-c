/**
 * test_take.c - Tests for subregion extraction
 */

#include "test_common.h"

void test_ndarray_take_axis0_2d(void) {
    size_t dims[] = {4, 5, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 20.0, 1.0);
    
    // Take rows 1:3
    NDArray B = ndarray_new_take(A, 0, 1, 3);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 2);
    CU_ASSERT_EQUAL(B->dims[0], 2);  // rows 1 and 2
    CU_ASSERT_EQUAL(B->dims[1], 5);
    
    // Check first row (row 1 of A)
    CU_ASSERT_DOUBLE_EQUAL(B->data[0], 5.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[1], 6.0, EPSILON);
    
    ndarray_free_all(NDA_LIST(A, B));
}

void test_ndarray_take_axis1_2d(void) {
    size_t dims[] = {3, 6, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 18.0, 1.0);
    
    // Take columns 2:5
    NDArray B = ndarray_new_take(A, 1, 2, 5);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 2);
    CU_ASSERT_EQUAL(B->dims[0], 3);
    CU_ASSERT_EQUAL(B->dims[1], 3);  // columns 2, 3, 4
    
    ndarray_free_all(NDA_LIST(A, B));
}

void test_ndarray_take_3d(void) {
    size_t dims[] = {2, 4, 3, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 24.0, 1.0);
    
    // Take along axis 1, indices 1:3
    NDArray B = ndarray_new_take(A, 1, 1, 3);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 3);
    CU_ASSERT_EQUAL(B->dims[0], 2);
    CU_ASSERT_EQUAL(B->dims[1], 2);  // indices 1 and 2
    CU_ASSERT_EQUAL(B->dims[2], 3);
    
    ndarray_free_all(NDA_LIST(A, B));
}

void test_ndarray_take_single_element(void) {
    size_t dims[] = {5, 3, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 15.0, 1.0);
    
    // Take single row (row 2), indices 2:3
    NDArray B = ndarray_new_take(A, 0, 2, 3);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 2);
    CU_ASSERT_EQUAL(B->dims[0], 1);
    CU_ASSERT_EQUAL(B->dims[1], 3);
    
    CU_ASSERT_DOUBLE_EQUAL(B->data[0], 6.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[1], 7.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[2], 8.0, EPSILON);
    
    ndarray_free_all(NDA_LIST(A, B));
}

/* Register tests for this module */
void register_take_tests(CU_pSuite suite) {
    CU_add_test(suite, "test take axis 0 2D", test_ndarray_take_axis0_2d);
    CU_add_test(suite, "test take axis 1 2D", test_ndarray_take_axis1_2d);
    CU_add_test(suite, "test take 3D", test_ndarray_take_3d);
    CU_add_test(suite, "test take single element", test_ndarray_take_single_element);
}
