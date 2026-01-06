/**
 * test_stack.c - Tests for stack
 */

#include "test_common.h"

void test_ndarray_stack_axis0_2d(void) {
    // Stack two [2, 3] arrays along axis 0 -> [2, 2, 3]
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new(dims);
    NDArray B = ndarray_new(dims);
    
    for (size_t i = 0; i < 6; ++i) {
        A->data[i] = i + 1.0;
        B->data[i] = (i + 1.0) * 10.0;
    }
    
    NDArray C = ndarray_new_stack(0, NDA_LIST(A, B));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 3);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 2);
    CU_ASSERT_EQUAL(C->dims[2], 3);
    
    // First array data
    CU_ASSERT_DOUBLE_EQUAL(C->data[0], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[5], 6.0, EPSILON);
    
    // Second array data
    CU_ASSERT_DOUBLE_EQUAL(C->data[6], 10.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[11], 60.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_stack_axis1_2d(void) {
    // Stack two [2, 3] arrays along axis 1 -> [2, 2, 3]
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_ones(dims);
    NDArray B = ndarray_new_full(dims, 2.0);
    
    NDArray C = ndarray_new_stack(1, NDA_LIST(A, B));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 3);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 2);
    CU_ASSERT_EQUAL(C->dims[2], 3);
    
    // Check pattern: should interleave by rows
    size_t pos1[] = {0, 0, 0};
    size_t pos2[] = {0, 1, 0};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(C, pos1), 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(C, pos2), 2.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_stack_axis2_2d(void) {
    // Stack three [2, 3] arrays along axis 2 -> [2, 3, 3]
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 1.0);
    NDArray B = ndarray_new_full(dims, 2.0);
    NDArray C_in = ndarray_new_full(dims, 3.0);
    
    NDArray C = ndarray_new_stack(2, NDA_LIST(A, B, C_in));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 3);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 3);
    CU_ASSERT_EQUAL(C->dims[2], 3);
    
    // Check values
    size_t pos1[] = {0, 0, 0};
    size_t pos2[] = {0, 0, 1};
    size_t pos3[] = {0, 0, 2};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(C, pos1), 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(C, pos2), 2.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(C, pos3), 3.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C_in);
    ndarray_free(C);
}

void test_ndarray_stack_3d(void) {
    // Stack two [2, 2, 3] arrays along axis 1 -> [2, 2, 2, 3]
    size_t dims[] = {2, 2, 3, 0};
    NDArray A = ndarray_new_ones(dims);
    NDArray B = ndarray_new_full(dims, 5.0);
    
    NDArray C = ndarray_new_stack(1, NDA_LIST(A, B));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 4);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 2);
    CU_ASSERT_EQUAL(C->dims[2], 2);
    CU_ASSERT_EQUAL(C->dims[3], 3);
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

/* Register tests for this module */
void register_stack_tests(CU_pSuite suite) {
    CU_add_test(suite, "test stack axis0 2d", test_ndarray_stack_axis0_2d);
    CU_add_test(suite, "test stack axis1 2d", test_ndarray_stack_axis1_2d);
    CU_add_test(suite, "test stack axis2 2d", test_ndarray_stack_axis2_2d);
    CU_add_test(suite, "test stack 3d", test_ndarray_stack_3d);
}
