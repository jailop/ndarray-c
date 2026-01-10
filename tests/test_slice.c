/**
 * test_slice.c - Tests for slice access
 */

#include "test_common.h"

void test_ndarray_get_slice_ptr_2d(void) {
    size_t dims[] = {3, 4, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 12.0, 1.0);
    
    // Get pointer to row 1
    double* row1 = ndarray_get_slice_ptr(A, 0, 1);
    CU_ASSERT_DOUBLE_EQUAL(row1[0], 4.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(row1[1], 5.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(row1[2], 6.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(row1[3], 7.0, EPSILON);
    
    ndarray_free(A);
}

void test_ndarray_copy_slice_2d(void) {
    size_t dims[] = {3, 4, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 12.0, 1.0);
    NDArray B = ndarray_new_zeros(dims);
    
    // Copy row 1 from A to row 2 of B
    NDArray result = ndarray_copy_slice(B, 0, 2, A, 0, 1);
    CU_ASSERT_PTR_EQUAL(result, B);
    
    CU_ASSERT_DOUBLE_EQUAL(B->data[8], 4.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[9], 5.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[10], 6.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[11], 7.0, EPSILON);
    
    // Other rows should still be zero
    CU_ASSERT_DOUBLE_EQUAL(B->data[0], 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[4], 0.0, EPSILON);
    
    ndarray_free_all(NDA_LIST(A, B));
}

void test_ndarray_get_slice_size(void) {
    size_t dims2d[] = {3, 4, 0};
    NDArray A2d = ndarray_new(dims2d);
    
    CU_ASSERT_EQUAL(ndarray_get_slice_size(A2d, 0), 4);   // Row has 4 elements
    CU_ASSERT_EQUAL(ndarray_get_slice_size(A2d, 1), 3);   // Column has 3 elements
    
    size_t dims3d[] = {2, 3, 4, 0};
    NDArray A3d = ndarray_new(dims3d);
    
    CU_ASSERT_EQUAL(ndarray_get_slice_size(A3d, 0), 12);  // 3*4
    CU_ASSERT_EQUAL(ndarray_get_slice_size(A3d, 1), 8);   // 2*4
    CU_ASSERT_EQUAL(ndarray_get_slice_size(A3d, 2), 6);   // 2*3
    
    ndarray_free_all(NDA_LIST(A2d, A3d));
}

/* Register tests for this module */
void register_slice_tests(CU_pSuite suite) {
    CU_add_test(suite, "test get slice ptr 2d", test_ndarray_get_slice_ptr_2d);
    CU_add_test(suite, "test copy slice 2d", test_ndarray_copy_slice_2d);
    CU_add_test(suite, "test get slice size", test_ndarray_get_slice_size);
}
