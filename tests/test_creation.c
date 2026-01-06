/**
 * test_creation.c - Tests for array creation
 */

#include "test_common.h"

void test_ndarray_new_2d(void) {
    size_t dims[] = {3, 4, 0};
    NDArray arr = ndarray_new(dims);
    
    CU_ASSERT_PTR_NOT_NULL(arr);
    CU_ASSERT_EQUAL(arr->ndim, 2);
    CU_ASSERT_EQUAL(arr->dims[0], 3);
    CU_ASSERT_EQUAL(arr->dims[1], 4);
    
    ndarray_free(arr);
}

void test_ndarray_new_3d(void) {
    size_t dims[] = {2, 3, 4, 0};
    NDArray arr = ndarray_new(dims);
    
    CU_ASSERT_PTR_NOT_NULL(arr);
    CU_ASSERT_EQUAL(arr->ndim, 3);
    CU_ASSERT_EQUAL(arr->dims[0], 2);
    CU_ASSERT_EQUAL(arr->dims[1], 3);
    CU_ASSERT_EQUAL(arr->dims[2], 4);
    
    ndarray_free(arr);
}

void test_ndarray_new_4d(void) {
    size_t dims[] = {2, 3, 4, 5, 0};
    NDArray arr = ndarray_new(dims);
    
    CU_ASSERT_PTR_NOT_NULL(arr);
    CU_ASSERT_EQUAL(arr->ndim, 4);
    CU_ASSERT_EQUAL(arr->dims[0], 2);
    CU_ASSERT_EQUAL(arr->dims[1], 3);
    CU_ASSERT_EQUAL(arr->dims[2], 4);
    CU_ASSERT_EQUAL(arr->dims[3], 5);
    
    ndarray_free(arr);
}

void test_ndarray_new_zeros(void) {
    size_t dims[] = {2, 3, 0};
    NDArray arr = ndarray_new_zeros(dims);
    
    CU_ASSERT_PTR_NOT_NULL(arr);
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(arr->data[i], 0.0, EPSILON);
    }
    
    ndarray_free(arr);
}

void test_ndarray_new_ones(void) {
    size_t dims[] = {2, 3, 0};
    NDArray arr = ndarray_new_ones(dims);
    
    CU_ASSERT_PTR_NOT_NULL(arr);
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(arr->data[i], 1.0, EPSILON);
    }
    
    ndarray_free(arr);
}

void test_ndarray_new_full(void) {
    size_t dims[] = {2, 3, 0};
    NDArray arr = ndarray_new_full(dims, 3.14);
    
    CU_ASSERT_PTR_NOT_NULL(arr);
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(arr->data[i], 3.14, EPSILON);
    }
    
    ndarray_free(arr);
}

void test_ndarray_new_arange(void) {
    size_t dims[] = {2, 5, 0};
    NDArray arr = ndarray_new_arange(dims, 0.0, 10.0, 1.0);
    
    CU_ASSERT_PTR_NOT_NULL(arr);
    for (size_t i = 0; i < 10; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(arr->data[i], (double)i, EPSILON);
    }
    
    ndarray_free(arr);
}

void test_ndarray_new_linspace(void) {
    size_t dims[] = {1, 5, 0};
    NDArray arr = ndarray_new_linspace(dims, 0.0, 4.0, 5);
    
    CU_ASSERT_PTR_NOT_NULL(arr);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[0], 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[1], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[2], 2.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[3], 3.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(arr->data[4], 4.0, EPSILON);
    
    ndarray_free(arr);
}

/* Register tests for this module */
void register_creation_tests(CU_pSuite suite) {
    CU_add_test(suite, "test new 2d", test_ndarray_new_2d);
    CU_add_test(suite, "test new 3d", test_ndarray_new_3d);
    CU_add_test(suite, "test new 4d", test_ndarray_new_4d);
    CU_add_test(suite, "test new zeros", test_ndarray_new_zeros);
    CU_add_test(suite, "test new ones", test_ndarray_new_ones);
    CU_add_test(suite, "test new full", test_ndarray_new_full);
    CU_add_test(suite, "test new arange", test_ndarray_new_arange);
    CU_add_test(suite, "test new linspace", test_ndarray_new_linspace);
}
