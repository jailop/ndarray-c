/**
 * test_reshape.c - Tests for reshape
 */

#include "test_common.h"

void test_ndarray_reshape_2d_to_2d(void) {
    size_t dims[] = {2, 3, 0};
    NDArray arr = ndarray_new(dims);
    
    // Fill with test data
    for (int i = 0; i < 6; i++) {
        arr->data[i] = i + 1;
    }
    
    // Reshape [2,3] -> [3,2]
    size_t new_dims[] = {3, 2, 0};
    ndarray_reshape(arr, new_dims);
    
    CU_ASSERT_EQUAL(arr->ndim, 2);
    CU_ASSERT_EQUAL(arr->dims[0], 3);
    CU_ASSERT_EQUAL(arr->dims[1], 2);
    
    // Verify data order is preserved (row-major)
    CU_ASSERT_DOUBLE_EQUAL(arr->data[0], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[1], 2.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[2], 3.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[3], 4.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[4], 5.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[5], 6.0, EPSILON);
    
    ndarray_free(arr);
}

void test_ndarray_reshape_2d_to_3d(void) {
    size_t dims[] = {2, 6, 0};
    NDArray arr = ndarray_new(dims);
    
    // Fill with test data
    for (int i = 0; i < 12; i++) {
        arr->data[i] = i * 10;
    }
    
    // Reshape [2,6] -> [2,3,2]
    size_t new_dims[] = {2, 3, 2, 0};
    ndarray_reshape(arr, new_dims);
    
    CU_ASSERT_EQUAL(arr->ndim, 3);
    CU_ASSERT_EQUAL(arr->dims[0], 2);
    CU_ASSERT_EQUAL(arr->dims[1], 3);
    CU_ASSERT_EQUAL(arr->dims[2], 2);
    
    // Verify data preserved
    CU_ASSERT_DOUBLE_EQUAL(arr->data[0], 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[11], 110.0, EPSILON);
    
    ndarray_free(arr);
}

void test_ndarray_reshape_3d_to_2d(void) {
    size_t dims[] = {2, 1, 4, 0};
    NDArray arr = ndarray_new(dims);
    
    // Fill with test data
    for (int i = 0; i < 8; i++) {
        arr->data[i] = i + 1;
    }
    
    // Reshape [2,1,4] -> [2,4] (squeeze middle dimension)
    size_t new_dims[] = {2, 4, 0};
    ndarray_reshape(arr, new_dims);
    
    CU_ASSERT_EQUAL(arr->ndim, 2);
    CU_ASSERT_EQUAL(arr->dims[0], 2);
    CU_ASSERT_EQUAL(arr->dims[1], 4);
    
    // Verify data preserved
    for (int i = 0; i < 8; i++) {
        CU_ASSERT_DOUBLE_EQUAL(arr->data[i], (double)(i + 1), EPSILON);
    }
    
    ndarray_free(arr);
}

void test_ndarray_reshape_with_inferred_dim(void) {
    size_t dims[] = {2, 6, 0};
    NDArray arr = ndarray_new(dims);
    
    // Fill with test data
    for (int i = 0; i < 12; i++) {
        arr->data[i] = i;
    }
    
    // Reshape [2,6] -> [3,-1] should infer -1 as 4
    size_t new_dims[] = {3, (size_t)-1, 0};
    ndarray_reshape(arr, new_dims);
    
    CU_ASSERT_EQUAL(arr->ndim, 2);
    CU_ASSERT_EQUAL(arr->dims[0], 3);
    CU_ASSERT_EQUAL(arr->dims[1], 4);
    
    // Verify data preserved
    CU_ASSERT_DOUBLE_EQUAL(arr->data[0], 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[11], 11.0, EPSILON);
    
    ndarray_free(arr);
}

void test_ndarray_reshape_flatten(void) {
    size_t dims[] = {2, 3, 4, 0};
    NDArray arr = ndarray_new(dims);
    
    // Fill with test data
    for (int i = 0; i < 24; i++) {
        arr->data[i] = i;
    }
    
    // Flatten [2,3,4] -> [1,24]
    size_t new_dims[] = {1, 24, 0};
    ndarray_reshape(arr, new_dims);
    
    CU_ASSERT_EQUAL(arr->ndim, 2);
    CU_ASSERT_EQUAL(arr->dims[0], 1);
    CU_ASSERT_EQUAL(arr->dims[1], 24);
    
    // Verify all data preserved
    for (int i = 0; i < 24; i++) {
        CU_ASSERT_DOUBLE_EQUAL(arr->data[i], (double)i, EPSILON);
    }
    
    ndarray_free(arr);
}

void test_ndarray_reshape_4d(void) {
    size_t dims[] = {6, 4, 0};
    NDArray arr = ndarray_new(dims);
    
    // Fill with test data
    for (int i = 0; i < 24; i++) {
        arr->data[i] = i + 1;
    }
    
    // Reshape [6,4] -> [2,3,2,2]
    size_t new_dims[] = {2, 3, 2, 2, 0};
    ndarray_reshape(arr, new_dims);
    
    CU_ASSERT_EQUAL(arr->ndim, 4);
    CU_ASSERT_EQUAL(arr->dims[0], 2);
    CU_ASSERT_EQUAL(arr->dims[1], 3);
    CU_ASSERT_EQUAL(arr->dims[2], 2);
    CU_ASSERT_EQUAL(arr->dims[3], 2);
    
    // Verify data preserved
    CU_ASSERT_DOUBLE_EQUAL(arr->data[0], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[23], 24.0, EPSILON);
    
    ndarray_free(arr);
}

/* Register tests for this module */
void register_reshape_tests(CU_pSuite suite) {
    CU_add_test(suite, "test reshape 2d to 2d", test_ndarray_reshape_2d_to_2d);
    CU_add_test(suite, "test reshape 2d to 3d", test_ndarray_reshape_2d_to_3d);
    CU_add_test(suite, "test reshape 3d to 2d", test_ndarray_reshape_3d_to_2d);
    CU_add_test(suite, "test reshape with inferred dim", test_ndarray_reshape_with_inferred_dim);
    CU_add_test(suite, "test reshape flatten", test_ndarray_reshape_flatten);
    CU_add_test(suite, "test reshape 4d", test_ndarray_reshape_4d);
}
