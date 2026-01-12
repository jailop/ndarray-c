/**
 * test_operations.c - Tests for array operations
 */

#include "test_common.h"

void test_ndarray_get_set_2d(void) {
    size_t dims[] = {3, 4, 0};
    NDArray arr = ndarray_new(dims);
    
    size_t pos[] = {1, 2};
    ndarray_set(arr, pos, 42.0);
    double val = ndarray_get(arr, pos);
    
    CU_ASSERT_DOUBLE_EQUAL(val, 42.0, EPSILON);
    
    ndarray_free(arr);
}

void test_ndarray_get_set_3d(void) {
    size_t dims[] = {2, 3, 4, 0};
    NDArray arr = ndarray_new(dims);
    
    size_t pos[] = {1, 2, 3};
    ndarray_set(arr, pos, 99.5);
    double val = ndarray_get(arr, pos);
    
    CU_ASSERT_DOUBLE_EQUAL(val, 99.5, EPSILON);
    
    ndarray_free(arr);
}

void test_ndarray_get_set_4d(void) {
    size_t dims[] = {2, 3, 4, 5, 0};
    NDArray arr = ndarray_new(dims);
    
    size_t pos[] = {1, 2, 3, 4};
    ndarray_set(arr, pos, -7.25);
    double val = ndarray_get(arr, pos);
    
    CU_ASSERT_DOUBLE_EQUAL(val, -7.25, EPSILON);
    
    ndarray_free(arr);
}

void test_ndarray_set_slice_2d_row(void) {
    size_t dims[] = {3, 4, 0};
    NDArray arr = ndarray_new_zeros(dims);
    
    // Set row 1 to [10, 20, 30, 40]
    double row_values[] = {10.0, 20.0, 30.0, 40.0};
    ndarray_set_slice(arr, 0, 1, row_values);
    
    // Check row 1
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(1, 0)), 10.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(1, 1)), 20.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(1, 2)), 30.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(1, 3)), 40.0, EPSILON);
    
    // Check other rows are still zero
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 0)), 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(2, 0)), 0.0, EPSILON);
    
    ndarray_free(arr);
}

void test_ndarray_set_slice_2d_col(void) {
    size_t dims[] = {3, 4, 0};
    NDArray arr = ndarray_new_zeros(dims);
    
    // Set column 2 to [5, 15, 25]
    double col_values[] = {5.0, 15.0, 25.0};
    ndarray_set_slice(arr, 1, 2, col_values);
    
    // Check column 2
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 2)), 5.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(1, 2)), 15.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(2, 2)), 25.0, EPSILON);
    
    // Check other columns are still zero
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 0)), 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 3)), 0.0, EPSILON);
    
    ndarray_free(arr);
}

void test_ndarray_set_slice_3d(void) {
    size_t dims[] = {2, 3, 4, 0};
    NDArray arr = ndarray_new_zeros(dims);
    
    // Set middle plane (axis=1, index=1) to sequential values
    double plane_values[8];  // 2*4 = 8 values
    for (int i = 0; i < 8; i++) {
        plane_values[i] = i * 10.0;
    }
    ndarray_set_slice(arr, 1, 1, plane_values);
    
    // Check the plane was set correctly
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 1, 0)), 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 1, 1)), 10.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(1, 1, 2)), 60.0, EPSILON);
    
    // Check other planes are still zero
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 0, 0)), 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 2, 0)), 0.0, EPSILON);
    
    ndarray_free(arr);
}

void test_ndarray_fill_slice_2d_row(void) {
    size_t dims[] = {3, 4, 0};
    NDArray arr = ndarray_new_zeros(dims);
    
    // Fill row 2 with 99.0
    ndarray_fill_slice(arr, 0, 2, 99.0);
    
    // Check row 2
    for (int j = 0; j < 4; j++) {
        CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(2, j)), 99.0, EPSILON);
    }
    
    // Check other rows are still zero
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 0)), 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(1, 0)), 0.0, EPSILON);
    
    ndarray_free(arr);
}

void test_ndarray_fill_slice_2d_col(void) {
    size_t dims[] = {3, 4, 0};
    NDArray arr = ndarray_new_zeros(dims);
    
    // Fill column 0 with -5.5
    ndarray_fill_slice(arr, 1, 0, -5.5);
    
    // Check column 0
    for (int i = 0; i < 3; i++) {
        CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(i, 0)), -5.5, EPSILON);
    }
    
    // Check other columns are still zero
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 1)), 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 3)), 0.0, EPSILON);
    
    ndarray_free(arr);
}

void test_ndarray_fill_slice_4d(void) {
    size_t dims[] = {2, 3, 4, 5, 0};
    NDArray arr = ndarray_new_zeros(dims);
    
    // Fill a hyperplane (axis=2, index=1) with 42.0
    ndarray_fill_slice(arr, 2, 1, 42.0);
    
    // Check some values in the hyperplane
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 0, 1, 0)), 42.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(1, 2, 1, 4)), 42.0, EPSILON);
    
    // Check other hyperplanes are still zero
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 0, 0, 0)), 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 0, 2, 0)), 0.0, EPSILON);
    
    ndarray_free(arr);
}

void test_ndarray_copy_2d(void) {
    size_t dims[] = {2, 3, 0};
    NDArray arr = ndarray_new_full(dims, 5.5);
    NDArray copy = ndarray_new_copy(arr);
    
    CU_ASSERT_PTR_NOT_NULL(copy);
    CU_ASSERT_EQUAL(copy->ndim, arr->ndim);
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(copy->data[i], arr->data[i], EPSILON);
    }
    
    ndarray_free(arr);
    ndarray_free(copy);
}

void test_ndarray_copy_3d(void) {
    size_t dims[] = {2, 3, 4, 0};
    NDArray arr = ndarray_new_arange(dims, 0.0, 24.0, 1.0);
    NDArray copy = ndarray_new_copy(arr);
    
    CU_ASSERT_PTR_NOT_NULL(copy);
    CU_ASSERT_EQUAL(copy->ndim, arr->ndim);
    for (size_t i = 0; i < 24; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(copy->data[i], arr->data[i], EPSILON);
    }
    
    ndarray_free(arr);
    ndarray_free(copy);
}

void test_ndarray_slice_chaining(void) {
    size_t dims[] = {3, 4, 0};
    NDArray arr = ndarray_new_zeros(dims);
    
    // Test chaining: set first row, then fill second row
    double row_data[] = {1.0, 2.0, 3.0, 4.0};
    ndarray_fill_slice(
        ndarray_set_slice(arr, 0, 0, row_data),
        0, 1, 5.0
    );
    
    // Check first row was set
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 0)), 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 1)), 2.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 2)), 3.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(0, 3)), 4.0, EPSILON);
    
    // Check second row was filled
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(1, 0)), 5.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(1, 1)), 5.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(1, 2)), 5.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(1, 3)), 5.0, EPSILON);
    
    // Check third row is still zero
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(arr, NDA_POS(2, 0)), 0.0, EPSILON);
    
    ndarray_free(arr);
}

/* Register tests for this module */
void register_operations_tests(CU_pSuite suite) {
    CU_add_test(suite, "test get set 2d", test_ndarray_get_set_2d);
    CU_add_test(suite, "test get set 3d", test_ndarray_get_set_3d);
    CU_add_test(suite, "test get set 4d", test_ndarray_get_set_4d);
    CU_add_test(suite, "test set slice 2d row", test_ndarray_set_slice_2d_row);
    CU_add_test(suite, "test set slice 2d col", test_ndarray_set_slice_2d_col);
    CU_add_test(suite, "test set slice 3d", test_ndarray_set_slice_3d);
    CU_add_test(suite, "test fill slice 2d row", test_ndarray_fill_slice_2d_row);
    CU_add_test(suite, "test fill slice 2d col", test_ndarray_fill_slice_2d_col);
    CU_add_test(suite, "test fill slice 4d", test_ndarray_fill_slice_4d);
    CU_add_test(suite, "test slice chaining", test_ndarray_slice_chaining);
    CU_add_test(suite, "test copy 2d", test_ndarray_copy_2d);
    CU_add_test(suite, "test copy 3d", test_ndarray_copy_3d);
}
