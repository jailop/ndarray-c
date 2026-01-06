/**
 * test_concat.c - Tests for concatenate
 */

#include "test_common.h"

void test_ndarray_concat_axis0_2d(void) {
    // Concatenate along first axis: [2,3] + [3,3] -> [5,3]
    NDArray A = ndarray_new_ones(NDA_DIMS(2, 3));
    NDArray B = ndarray_new_full(NDA_DIMS(3, 3), 2.0);
    
    NDArray C = ndarray_new_concat(0, NDA_LIST(A, B));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 2);
    CU_ASSERT_EQUAL(C->dims[0], 5);
    CU_ASSERT_EQUAL(C->dims[1], 3);
    
    // Check values
    CU_ASSERT_DOUBLE_EQUAL(C->data[0], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[6], 2.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_concat_axis1_2d(void) {
    // Concatenate along second axis: [2,3] + [2,5] -> [2,8]
    NDArray A = ndarray_new_ones(NDA_DIMS(2, 3));
    NDArray B = ndarray_new_full(NDA_DIMS(2, 5), 2.0);
    
    NDArray C = ndarray_new_concat(1, NDA_LIST(A, B));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 2);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 8);
    
    // Check values at boundary
    size_t pos1[] = {0, 2};
    size_t pos2[] = {0, 3};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(C, pos1), 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(C, pos2), 2.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_concat_3d_middle(void) {
    // Concatenate along middle axis: [2,3,4] + [2,5,4] -> [2,8,4]
    NDArray A = ndarray_new_ones(NDA_DIMS(2, 3, 4));
    NDArray B = ndarray_new_full(NDA_DIMS(2, 5, 4), 3.0);
    
    NDArray C = ndarray_new_concat(1, NDA_LIST(A, B));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 3);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 8);
    CU_ASSERT_EQUAL(C->dims[2], 4);
    
    // Check values
    size_t pos1[] = {0, 2, 0};
    size_t pos2[] = {0, 3, 0};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(C, pos1), 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(C, pos2), 3.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_concat_multiple(void) {
    // Concatenate three arrays: [1,10] + [1,20] + [1,30] -> [1,60]
    NDArray A = ndarray_new_ones(NDA_DIMS(1, 10));
    NDArray B = ndarray_new_full(NDA_DIMS(1, 20), 2.0);
    NDArray D = ndarray_new_full(NDA_DIMS(1, 30), 3.0);
    
    NDArray C = ndarray_new_concat(1, NDA_LIST(A, B, D));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 2);
    CU_ASSERT_EQUAL(C->dims[0], 1);
    CU_ASSERT_EQUAL(C->dims[1], 60);
    
    // Check values
    CU_ASSERT_DOUBLE_EQUAL(C->data[9], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[15], 2.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[35], 3.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(D);
    ndarray_free(C);
}

/* Register tests for this module */
void register_concat_tests(CU_pSuite suite) {
    CU_add_test(suite, "test concat axis0 2d", test_ndarray_concat_axis0_2d);
    CU_add_test(suite, "test concat axis1 2d", test_ndarray_concat_axis1_2d);
    CU_add_test(suite, "test concat 3d middle", test_ndarray_concat_3d_middle);
    CU_add_test(suite, "test concat multiple", test_ndarray_concat_multiple);
}
