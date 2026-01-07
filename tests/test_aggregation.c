/**
 * test_aggregation.c - Tests for aggregations
 */

#include "test_common.h"

void test_ndarray_aggr_sum_all_2d(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_arange(dims, 1.0, 7.0, 1.0);
    NDArray result = ndarray_new_aggr(A, NDA_ALL_AXES, NDA_AGGR_SUM);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 21.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(result);
}

void test_ndarray_aggr_mean_all_3d(void) {
    size_t dims[] = {2, 2, 2, 0};
    NDArray A = ndarray_new_full(dims, 4.0);
    NDArray result = ndarray_new_aggr(A, NDA_ALL_AXES, NDA_AGGR_MEAN);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 4.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(result);
}

void test_ndarray_aggr_max_all_4d(void) {
    size_t dims[] = {2, 2, 2, 2, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 16.0, 1.0);
    NDArray result = ndarray_new_aggr(A, NDA_ALL_AXES, NDA_AGGR_MAX);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 15.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(result);
}

void test_ndarray_aggr_min_all_2d(void) {
    size_t dims[] = {3, 3, 0};
    NDArray A = ndarray_new_arange(dims, 5.0, 14.0, 1.0);
    NDArray result = ndarray_new_aggr(A, NDA_ALL_AXES, NDA_AGGR_MIN);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 5.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(result);
}

void test_ndarray_aggr_sum_axis0_2d(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new(dims);
    // A = [[1, 2, 3], [4, 5, 6]]
    for (size_t i = 0; i < 6; ++i) {
        A->data[i] = i + 1;
    }
    
    NDArray result = ndarray_new_aggr(A, 0, NDA_AGGR_SUM);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_EQUAL(result->dims[0], 1);
    CU_ASSERT_EQUAL(result->dims[1], 3);
    
    // [5, 7, 9] -> reshaped to [1, 3]
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 5.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(result->data[1], 7.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(result->data[2], 9.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(result);
}

void test_ndarray_aggr_sum_axis1_2d(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new(dims);
    // A = [[1, 2, 3], [4, 5, 6]]
    for (size_t i = 0; i < 6; ++i) {
        A->data[i] = i + 1;
    }
    
    NDArray result = ndarray_new_aggr(A, 1, NDA_AGGR_SUM);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_EQUAL(result->dims[0], 2);
    CU_ASSERT_EQUAL(result->dims[1], 1);
    
    // [6, 15] -> reshaped to [2, 1]
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 6.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(result->data[1], 15.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(result);
}

void test_ndarray_aggr_mean_axis0_3d(void) {
    size_t dims[] = {2, 3, 4, 0};
    NDArray A = ndarray_new_full(dims, 6.0);
    NDArray result = ndarray_new_aggr(A, 0, NDA_AGGR_MEAN);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_EQUAL(result->dims[0], 3);
    CU_ASSERT_EQUAL(result->dims[1], 4);
    
    for (size_t i = 0; i < 12; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(result->data[i], 6.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(result);
}

void test_ndarray_aggr_max_axis2_3d(void) {
    size_t dims[] = {2, 3, 4, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 24.0, 1.0);
    NDArray result = ndarray_new_aggr(A, 2, NDA_AGGR_MAX);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_EQUAL(result->dims[0], 2);
    CU_ASSERT_EQUAL(result->dims[1], 3);
    
    // Each row's max should be 3, 7, 11, 15, 19, 23
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 3.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(result->data[1], 7.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(result->data[2], 11.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(result->data[3], 15.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(result->data[4], 19.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(result->data[5], 23.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(result);
}

void test_ndarray_aggr_min_axis1_4d(void) {
    size_t dims[] = {2, 3, 2, 2, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 24.0, 1.0);
    NDArray result = ndarray_new_aggr(A, 1, NDA_AGGR_MIN);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_EQUAL(result->dims[0], 2);
    CU_ASSERT_EQUAL(result->dims[1], 2);
    CU_ASSERT_EQUAL(result->dims[2], 2);
    
    ndarray_free(A);
    ndarray_free(result);
}

/* Register tests for this module */
void register_aggregation_tests(CU_pSuite suite) {
    CU_add_test(suite, "test aggr sum all 2d", test_ndarray_aggr_sum_all_2d);
    CU_add_test(suite, "test aggr mean all 3d", test_ndarray_aggr_mean_all_3d);
    CU_add_test(suite, "test aggr max all 4d", test_ndarray_aggr_max_all_4d);
    CU_add_test(suite, "test aggr min all 2d", test_ndarray_aggr_min_all_2d);
    CU_add_test(suite, "test aggr sum axis0 2d", test_ndarray_aggr_sum_axis0_2d);
    CU_add_test(suite, "test aggr sum axis1 2d", test_ndarray_aggr_sum_axis1_2d);
    CU_add_test(suite, "test aggr mean axis0 3d", test_ndarray_aggr_mean_axis0_3d);
    CU_add_test(suite, "test aggr max axis2 3d", test_ndarray_aggr_max_axis2_3d);
    CU_add_test(suite, "test aggr min axis1 4d", test_ndarray_aggr_min_axis1_4d);
}
