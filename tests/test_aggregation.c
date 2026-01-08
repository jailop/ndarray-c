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

void test_ndarray_scalar_aggr_sum(void) {
    size_t dims[] = {3, 4, 0};
    NDArray A = ndarray_new_from_data(dims, 
        (double[]){1,2,3,4,5,6,7,8,9,10,11,12});
    
    double result = ndarray_scalar_aggr(A, NDA_AGGR_SUM);
    CU_ASSERT_DOUBLE_EQUAL(result, 78.0, EPSILON);
    
    ndarray_free(A);
}

void test_ndarray_scalar_aggr_mean(void) {
    size_t dims[] = {3, 4, 0};
    NDArray A = ndarray_new_from_data(dims, 
        (double[]){1,2,3,4,5,6,7,8,9,10,11,12});
    
    double result = ndarray_scalar_aggr(A, NDA_AGGR_MEAN);
    CU_ASSERT_DOUBLE_EQUAL(result, 6.5, EPSILON);
    
    ndarray_free(A);
}

void test_ndarray_scalar_aggr_max(void) {
    size_t dims[] = {3, 4, 0};
    NDArray A = ndarray_new_from_data(dims, 
        (double[]){1,2,3,4,5,6,7,8,9,10,11,12});
    
    double result = ndarray_scalar_aggr(A, NDA_AGGR_MAX);
    CU_ASSERT_DOUBLE_EQUAL(result, 12.0, EPSILON);
    
    ndarray_free(A);
}

void test_ndarray_scalar_aggr_min(void) {
    size_t dims[] = {3, 4, 0};
    NDArray A = ndarray_new_from_data(dims, 
        (double[]){1,2,3,4,5,6,7,8,9,10,11,12});
    
    double result = ndarray_scalar_aggr(A, NDA_AGGR_MIN);
    CU_ASSERT_DOUBLE_EQUAL(result, 1.0, EPSILON);
    
    ndarray_free(A);
}

void test_ndarray_scalar_aggr_std(void) {
    size_t dims[] = {2, 5, 0};
    NDArray A = ndarray_new_from_data(dims, 
        (double[]){2,4,4,4,5,5,5,7,9,9});
    
    double result = ndarray_scalar_aggr(A, NDA_AGGR_STD);
    // Expected std dev = 2.1540659228538015
    CU_ASSERT_DOUBLE_EQUAL(result, 2.1540659228538015, EPSILON);
    
    ndarray_free(A);
}

void test_ndarray_scalar_aggr_consistency(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_arange(dims, 1.0, 7.0, 1.0);
    
    // Test that scalar_aggr matches new_aggr with NDA_ALL_AXES
    NDArray result_array = ndarray_new_aggr(A, NDA_ALL_AXES, NDA_AGGR_SUM);
    double result_scalar = ndarray_scalar_aggr(A, NDA_AGGR_SUM);
    
    CU_ASSERT_DOUBLE_EQUAL(result_array->data[0], result_scalar, EPSILON);
    
    ndarray_free(A);
    ndarray_free(result_array);
}

void test_ndarray_scalar_aggr_negative_values(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_from_data(dims, 
        (double[]){-3, -1, 2, 4, -5, 6});
    
    double sum = ndarray_scalar_aggr(A, NDA_AGGR_SUM);
    double mean = ndarray_scalar_aggr(A, NDA_AGGR_MEAN);
    double max = ndarray_scalar_aggr(A, NDA_AGGR_MAX);
    double min = ndarray_scalar_aggr(A, NDA_AGGR_MIN);
    
    CU_ASSERT_DOUBLE_EQUAL(sum, 3.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(mean, 0.5, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(max, 6.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(min, -5.0, EPSILON);
    
    ndarray_free(A);
}

void test_ndarray_scalar_aggr_large_array(void) {
    size_t dims[] = {100, 100, 0};
    NDArray A = ndarray_new_full(dims, 2.5);
    
    double sum = ndarray_scalar_aggr(A, NDA_AGGR_SUM);
    double mean = ndarray_scalar_aggr(A, NDA_AGGR_MEAN);
    double max = ndarray_scalar_aggr(A, NDA_AGGR_MAX);
    double min = ndarray_scalar_aggr(A, NDA_AGGR_MIN);
    
    CU_ASSERT_DOUBLE_EQUAL(sum, 25000.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(mean, 2.5, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(max, 2.5, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(min, 2.5, EPSILON);
    
    ndarray_free(A);
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
    
    CU_add_test(suite, "test scalar aggr sum", test_ndarray_scalar_aggr_sum);
    CU_add_test(suite, "test scalar aggr mean", test_ndarray_scalar_aggr_mean);
    CU_add_test(suite, "test scalar aggr max", test_ndarray_scalar_aggr_max);
    CU_add_test(suite, "test scalar aggr min", test_ndarray_scalar_aggr_min);
    CU_add_test(suite, "test scalar aggr std", test_ndarray_scalar_aggr_std);
    CU_add_test(suite, "test scalar aggr consistency", test_ndarray_scalar_aggr_consistency);
    CU_add_test(suite, "test scalar aggr negative", test_ndarray_scalar_aggr_negative_values);
    CU_add_test(suite, "test scalar aggr large", test_ndarray_scalar_aggr_large_array);
}
