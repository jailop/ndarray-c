/**
 * ndarray_test.c - Comprehensive test suite for ndarray library
 * 
 * Tests all ndarray functions with 2D, 3D, and 4D arrays
 * Compile: gcc -fopenmp -o ndarray_test ndarray_test.c ndarray.c randnorm.c -lm -lcunit
 * Run: ./ndarray_test
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CUnit/Basic.h>
#include "ndarray.h"

#define EPSILON 1e-6

/* Test Suite Initialization */
int init_suite(void) {
    return 0;
}

int clean_suite(void) {
    return 0;
}

/* Helper function to compare doubles */
int double_equals(double a, double b) {
    return fabs(a - b) < EPSILON;
}

/* ========== Test: Array Creation ========== */

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

/* ========== Test: Array Operations ========== */

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

void test_ndarray_copy_2d(void) {
    size_t dims[] = {2, 3, 0};
    NDArray arr = ndarray_new_full(dims, 5.5);
    NDArray copy = ndarray_copy(arr);
    
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
    NDArray copy = ndarray_copy(arr);
    
    CU_ASSERT_PTR_NOT_NULL(copy);
    CU_ASSERT_EQUAL(copy->ndim, arr->ndim);
    for (size_t i = 0; i < 24; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(copy->data[i], arr->data[i], EPSILON);
    }
    
    ndarray_free(arr);
    ndarray_free(copy);
}

/* ========== Test: Arithmetic Operations ========== */

void test_ndarray_add_2d(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 2.0);
    NDArray B = ndarray_new_full(dims, 3.0);
    
    ndarray_add(A, B);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 5.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_add_3d(void) {
    size_t dims[] = {2, 2, 2, 0};
    NDArray A = ndarray_new_full(dims, 1.5);
    NDArray B = ndarray_new_full(dims, 2.5);
    
    ndarray_add(A, B);
    
    for (size_t i = 0; i < 8; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 4.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_mul_2d(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 2.0);
    NDArray B = ndarray_new_full(dims, 3.0);
    
    ndarray_mul(A, B);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 6.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_mul_3d(void) {
    size_t dims[] = {2, 2, 2, 0};
    NDArray A = ndarray_new_full(dims, 4.0);
    NDArray B = ndarray_new_full(dims, 0.5);
    
    ndarray_mul(A, B);
    
    for (size_t i = 0; i < 8; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 2.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_add_scalar(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 10.0);
    
    ndarray_add_scalar(A, 5.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 15.0, EPSILON);
    }
    
    ndarray_free(A);
}

void test_ndarray_mul_scalar(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 4.0);
    
    ndarray_mul_scalar(A, 2.5);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 10.0, EPSILON);
    }
    
    ndarray_free(A);
}

double square_func(double x) {
    return x * x;
}

void test_ndarray_mapfnc(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_arange(dims, 1.0, 7.0, 1.0);
    
    ndarray_mapfnc(A, square_func);
    
    CU_ASSERT_DOUBLE_EQUAL(A->data[0], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[1], 4.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[2], 9.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[3], 16.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[4], 25.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[5], 36.0, EPSILON);
    
    ndarray_free(A);
}

/* ========== Test: Matrix Multiplication ========== */

void test_ndarray_matmul_2d(void) {
    // A: 2x3, B: 3x2 -> C: 2x2
    size_t dims_a[] = {2, 3, 0};
    size_t dims_b[] = {3, 2, 0};
    
    NDArray A = ndarray_new(dims_a);
    NDArray B = ndarray_new(dims_b);
    
    // A = [[1, 2, 3], [4, 5, 6]]
    A->data[0] = 1; A->data[1] = 2; A->data[2] = 3;
    A->data[3] = 4; A->data[4] = 5; A->data[5] = 6;
    
    // B = [[1, 2], [3, 4], [5, 6]]
    B->data[0] = 1; B->data[1] = 2;
    B->data[2] = 3; B->data[3] = 4;
    B->data[4] = 5; B->data[5] = 6;
    
    NDArray C = ndarray_new_matmul(A, B);
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 2);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 2);
    
    // C = [[22, 28], [49, 64]]
    CU_ASSERT_DOUBLE_EQUAL(C->data[0], 22.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[1], 28.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[2], 49.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[3], 64.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_matmul_3d_batch(void) {
    // Batch of 2 matrices: [2, 2, 3] @ [2, 3, 2] -> [2, 2, 2]
    size_t dims_a[] = {2, 2, 3, 0};
    size_t dims_b[] = {2, 3, 2, 0};
    
    NDArray A = ndarray_new_ones(dims_a);
    NDArray B = ndarray_new_ones(dims_b);
    
    NDArray C = ndarray_new_matmul(A, B);
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 3);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 2);
    CU_ASSERT_EQUAL(C->dims[2], 2);
    
    // All elements should be 3.0 (sum of 3 ones)
    for (size_t i = 0; i < 8; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 3.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_matmul_4d_batch(void) {
    // Batch: [2, 3, 2, 2] @ [2, 3, 2, 2] -> [2, 3, 2, 2]
    size_t dims[] = {2, 3, 2, 2, 0};
    
    NDArray A = ndarray_new_full(dims, 2.0);
    NDArray B = ndarray_new_full(dims, 0.5);
    
    NDArray C = ndarray_new_matmul(A, B);
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 4);
    
    // Each 2x2 result should have all 2.0s (2*0.5 + 2*0.5 = 2.0)
    for (size_t i = 0; i < 24; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 2.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

/* ========== Test: Tensordot ========== */

void test_ndarray_tensordot_2d_matmul(void) {
    // Test tensordot equivalent to matrix multiplication
    // A: 2x3, B: 3x2 -> C: 2x2
    size_t dims_a[] = {2, 3, 0};
    size_t dims_b[] = {3, 2, 0};
    
    NDArray A = ndarray_new(dims_a);
    NDArray B = ndarray_new(dims_b);
    
    // A = [[1, 2, 3], [4, 5, 6]]
    A->data[0] = 1; A->data[1] = 2; A->data[2] = 3;
    A->data[3] = 4; A->data[4] = 5; A->data[5] = 6;
    
    // B = [[1, 2], [3, 4], [5, 6]]
    B->data[0] = 1; B->data[1] = 2;
    B->data[2] = 3; B->data[3] = 4;
    B->data[4] = 5; B->data[5] = 6;
    
    // Contract on last axis of A (axis 1) and first axis of B (axis 0)
    NDArray C = ndarray_new_tensordot(A, B, NDA_AXES(1), NDA_AXES(0));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 2);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 2);
    
    // C = [[22, 28], [49, 64]]
    CU_ASSERT_DOUBLE_EQUAL(C->data[0], 22.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[1], 28.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[2], 49.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(C->data[3], 64.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_tensordot_3d_single_axis(void) {
    // A: (2, 3, 4), B: (4, 5) -> C: (2, 3, 5)
    size_t dims_a[] = {2, 3, 4, 0};
    size_t dims_b[] = {4, 5, 0};
    
    NDArray A = ndarray_new_ones(dims_a);
    NDArray B = ndarray_new_full(dims_b, 2.0);
    
    // Contract on last axis of A (axis 2) and first axis of B (axis 0)
    NDArray C = ndarray_new_tensordot(A, B, NDA_AXES(2), NDA_AXES(0));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 3);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 3);
    CU_ASSERT_EQUAL(C->dims[2], 5);
    
    // All elements should be 8.0 (sum of 4 * 1.0 * 2.0)
    for (size_t i = 0; i < 30; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 8.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_tensordot_multi_axis(void) {
    // A: (2, 3, 4, 5), B: (4, 5, 6) -> contract axes (2,3) with (0,1) -> C: (2, 3, 6)
    size_t dims_a[] = {2, 3, 4, 5, 0};
    size_t dims_b[] = {4, 5, 6, 0};
    
    NDArray A = ndarray_new_ones(dims_a);
    NDArray B = ndarray_new_ones(dims_b);
    
    // Contract on axes 2,3 of A and axes 0,1 of B
    NDArray C = ndarray_new_tensordot(A, B, NDA_AXES(2, 3), NDA_AXES(0, 1));
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 3);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 3);
    CU_ASSERT_EQUAL(C->dims[2], 6);
    
    // All elements should be 20.0 (4*5 contractions of 1.0*1.0)
    for (size_t i = 0; i < 36; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 20.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

void test_ndarray_tensordot_vs_matmul(void) {
    // Verify tensordot and matmul produce same results for matrix multiplication
    size_t dims_a[] = {3, 4, 0};
    size_t dims_b[] = {4, 5, 0};
    
    NDArray A = ndarray_new(dims_a);
    NDArray B = ndarray_new(dims_b);
    
    // Fill with sequential values
    for (size_t i = 0; i < 12; ++i) A->data[i] = i + 1.0;
    for (size_t i = 0; i < 20; ++i) B->data[i] = (i + 1.0) * 0.5;
    
    // Compute using matmul
    NDArray C_matmul = ndarray_new_matmul(A, B);
    
    // Compute using tensordot
    NDArray C_tensordot = ndarray_new_tensordot(A, B, NDA_AXES(1), NDA_AXES(0));
    
    CU_ASSERT_PTR_NOT_NULL(C_matmul);
    CU_ASSERT_PTR_NOT_NULL(C_tensordot);
    CU_ASSERT_EQUAL(C_matmul->ndim, C_tensordot->ndim);
    CU_ASSERT_EQUAL(C_matmul->dims[0], C_tensordot->dims[0]);
    CU_ASSERT_EQUAL(C_matmul->dims[1], C_tensordot->dims[1]);
    
    // Compare all elements
    for (size_t i = 0; i < 15; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C_matmul->data[i], C_tensordot->data[i], EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C_matmul);
    ndarray_free(C_tensordot);
}

void test_ndarray_tensordot_outer_product(void) {
    // Test outer product: contract 0 axes -> outer product
    size_t dims_a[] = {2, 3, 0};
    size_t dims_b[] = {3, 4, 0};
    
    NDArray A = ndarray_new_full(dims_a, 2.0);
    NDArray B = ndarray_new_full(dims_b, 3.0);
    
    // No axes contracted -> outer product
    NDArray C = ndarray_new_tensordot(A, B, NDA_NOAXES, NDA_NOAXES);
    
    CU_ASSERT_PTR_NOT_NULL(C);
    CU_ASSERT_EQUAL(C->ndim, 4);
    CU_ASSERT_EQUAL(C->dims[0], 2);
    CU_ASSERT_EQUAL(C->dims[1], 3);
    CU_ASSERT_EQUAL(C->dims[2], 3);
    CU_ASSERT_EQUAL(C->dims[3], 4);
    
    // All elements should be 6.0 (2.0 * 3.0)
    for (size_t i = 0; i < 72; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 6.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

/* ========== Test: Stack ========== */

void test_ndarray_stack_axis0_2d(void) {
    // Stack two [2, 3] arrays along axis 0 -> [2, 2, 3]
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new(dims);
    NDArray B = ndarray_new(dims);
    
    for (size_t i = 0; i < 6; ++i) {
        A->data[i] = i + 1.0;
        B->data[i] = (i + 1.0) * 10.0;
    }
    
    NDArray C = ndarray_stack(0, NDA_LIST(A, B));
    
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
    
    NDArray C = ndarray_stack(1, NDA_LIST(A, B));
    
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
    
    NDArray C = ndarray_stack(2, NDA_LIST(A, B, C_in));
    
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
    
    NDArray C = ndarray_stack(1, NDA_LIST(A, B));
    
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

/* ========== Test: Concatenate ========== */

void test_ndarray_concat_axis0_2d(void) {
    // Concatenate along first axis: [2,3] + [3,3] -> [5,3]
    NDArray A = ndarray_new_ones(NDA_DIMS(2, 3));
    NDArray B = ndarray_new_full(NDA_DIMS(3, 3), 2.0);
    
    NDArray C = ndarray_concat(0, NDA_LIST(A, B));
    
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
    
    NDArray C = ndarray_concat(1, NDA_LIST(A, B));
    
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
    
    NDArray C = ndarray_concat(1, NDA_LIST(A, B));
    
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
    
    NDArray C = ndarray_concat(1, NDA_LIST(A, B, D));
    
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

/* ========== Test: Take (Subregion) ========== */

void test_ndarray_take_axis0_2d(void) {
    size_t dims[] = {4, 5, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 20.0, 1.0);
    
    // Take rows 1:3
    NDArray B = ndarray_take(A, 0, 1, 3);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 2);
    CU_ASSERT_EQUAL(B->dims[0], 2);
    CU_ASSERT_EQUAL(B->dims[1], 5);
    
    // First row should be [5, 6, 7, 8, 9]
    CU_ASSERT_DOUBLE_EQUAL(B->data[0], 5.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[4], 9.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_take_axis1_2d(void) {
    // Take columns from 2D array
    size_t dims[] = {4, 5, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 20.0, 1.0);
    
    // Take columns 1:4
    NDArray B = ndarray_take(A, 1, 1, 4);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 2);
    CU_ASSERT_EQUAL(B->dims[0], 4);
    CU_ASSERT_EQUAL(B->dims[1], 3);
    
    // First row should be [1, 2, 3]
    CU_ASSERT_DOUBLE_EQUAL(B->data[0], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[2], 3.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_take_3d(void) {
    // Take along middle axis in 3D array
    size_t dims[] = {2, 3, 4, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 24.0, 1.0);
    
    // Take axis 1, indices 1:3
    NDArray B = ndarray_take(A, 1, 1, 3);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 3);
    CU_ASSERT_EQUAL(B->dims[0], 2);
    CU_ASSERT_EQUAL(B->dims[1], 2);
    CU_ASSERT_EQUAL(B->dims[2], 4);
    
    // Check first element (should be from [0, 1, 0])
    size_t pos[] = {0, 0, 0};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(B, pos), 4.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_take_single_element(void) {
    // Take single column (axis 1)
    size_t dims[] = {4, 5, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 20.0, 1.0);
    
    // Take single column at index 2
    NDArray B = ndarray_take(A, 1, 2, 3);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 2);
    CU_ASSERT_EQUAL(B->dims[0], 4);
    CU_ASSERT_EQUAL(B->dims[1], 1);
    
    // Should contain [2, 7, 12, 17]
    CU_ASSERT_DOUBLE_EQUAL(B->data[0], 2.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[1], 7.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[2], 12.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[3], 17.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
}

/* ========== Test: Transpose ========== */

void test_ndarray_transpose_2d(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new(dims);
    
    // A = [[1, 2, 3], [4, 5, 6]]
    A->data[0] = 1; A->data[1] = 2; A->data[2] = 3;
    A->data[3] = 4; A->data[4] = 5; A->data[5] = 6;
    
    NDArray B = ndarray_new_transpose(A);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 2);
    CU_ASSERT_EQUAL(B->dims[0], 3);
    CU_ASSERT_EQUAL(B->dims[1], 2);
    
    // B = [[1, 4], [2, 5], [3, 6]]
    CU_ASSERT_DOUBLE_EQUAL(B->data[0], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[1], 4.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[2], 2.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[3], 5.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[4], 3.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(B->data[5], 6.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_transpose_3d(void) {
    size_t dims[] = {2, 3, 4, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 24.0, 1.0);
    NDArray B = ndarray_new_transpose(A);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 3);
    CU_ASSERT_EQUAL(B->dims[0], 4);
    CU_ASSERT_EQUAL(B->dims[1], 3);
    CU_ASSERT_EQUAL(B->dims[2], 2);
    
    // Check a few elements
    size_t pos_a[] = {0, 0, 0};
    size_t pos_b[] = {0, 0, 0};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(B, pos_b), ndarray_get(A, pos_a), EPSILON);
    
    size_t pos_a2[] = {1, 2, 3};
    size_t pos_b2[] = {3, 2, 1};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(B, pos_b2), ndarray_get(A, pos_a2), EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_transpose_4d(void) {
    size_t dims[] = {2, 3, 4, 5, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 120.0, 1.0);
    NDArray B = ndarray_new_transpose(A);
    
    CU_ASSERT_PTR_NOT_NULL(B);
    CU_ASSERT_EQUAL(B->ndim, 4);
    CU_ASSERT_EQUAL(B->dims[0], 5);
    CU_ASSERT_EQUAL(B->dims[1], 4);
    CU_ASSERT_EQUAL(B->dims[2], 3);
    CU_ASSERT_EQUAL(B->dims[3], 2);
    
    // Check corner elements
    size_t pos_a[] = {0, 0, 0, 0};
    size_t pos_b[] = {0, 0, 0, 0};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(B, pos_b), ndarray_get(A, pos_a), EPSILON);
    
    size_t pos_a2[] = {1, 2, 3, 4};
    size_t pos_b2[] = {4, 3, 2, 1};
    CU_ASSERT_DOUBLE_EQUAL(ndarray_get(B, pos_b2), ndarray_get(A, pos_a2), EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
}

/* ========== Test: Aggregations ========== */

void test_ndarray_aggr_sum_all_2d(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_arange(dims, 1.0, 7.0, 1.0);
    NDArray result = ndarray_new_axis_aggr(A, -1, NDARRAY_AGGR_SUM);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 21.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(result);
}

void test_ndarray_aggr_mean_all_3d(void) {
    size_t dims[] = {2, 2, 2, 0};
    NDArray A = ndarray_new_full(dims, 4.0);
    NDArray result = ndarray_new_axis_aggr(A, -1, NDARRAY_AGGR_MEAN);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 4.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(result);
}

void test_ndarray_aggr_max_all_4d(void) {
    size_t dims[] = {2, 2, 2, 2, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 16.0, 1.0);
    NDArray result = ndarray_new_axis_aggr(A, -1, NDARRAY_AGGR_MAX);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 15.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(result);
}

void test_ndarray_aggr_min_all_2d(void) {
    size_t dims[] = {3, 3, 0};
    NDArray A = ndarray_new_arange(dims, 5.0, 14.0, 1.0);
    NDArray result = ndarray_new_axis_aggr(A, -1, NDARRAY_AGGR_MIN);
    
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
    
    NDArray result = ndarray_new_axis_aggr(A, 0, NDARRAY_AGGR_SUM);
    
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
    
    NDArray result = ndarray_new_axis_aggr(A, 1, NDARRAY_AGGR_SUM);
    
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
    NDArray result = ndarray_new_axis_aggr(A, 0, NDARRAY_AGGR_MEAN);
    
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
    NDArray result = ndarray_new_axis_aggr(A, 2, NDARRAY_AGGR_MAX);
    
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
    NDArray result = ndarray_new_axis_aggr(A, 1, NDARRAY_AGGR_MIN);
    
    CU_ASSERT_PTR_NOT_NULL(result);
    CU_ASSERT_EQUAL(result->dims[0], 2);
    CU_ASSERT_EQUAL(result->dims[1], 2);
    CU_ASSERT_EQUAL(result->dims[2], 2);
    
    ndarray_free(A);
    ndarray_free(result);
}

/* ========== Main Test Runner ========== */

int main() {
    CU_pSuite suite_creation = NULL;
    CU_pSuite suite_operations = NULL;
    CU_pSuite suite_arithmetic = NULL;
    CU_pSuite suite_matmul = NULL;
    CU_pSuite suite_tensordot = NULL;
    CU_pSuite suite_stack = NULL;
    CU_pSuite suite_concat = NULL;
    CU_pSuite suite_take = NULL;
    CU_pSuite suite_transpose = NULL;
    CU_pSuite suite_aggregation = NULL;
    
    /* Initialize CUnit registry */
    if (CUE_SUCCESS != CU_initialize_registry()) {
        return CU_get_error();
    }
    
    /* Add suite: Array Creation */
    suite_creation = CU_add_suite("Array Creation", init_suite, clean_suite);
    if (NULL == suite_creation) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(suite_creation, "test ndarray_new 2D", test_ndarray_new_2d);
    CU_add_test(suite_creation, "test ndarray_new 3D", test_ndarray_new_3d);
    CU_add_test(suite_creation, "test ndarray_new 4D", test_ndarray_new_4d);
    CU_add_test(suite_creation, "test ndarray_new_zeros", test_ndarray_new_zeros);
    CU_add_test(suite_creation, "test ndarray_new_ones", test_ndarray_new_ones);
    CU_add_test(suite_creation, "test ndarray_new_full", test_ndarray_new_full);
    CU_add_test(suite_creation, "test ndarray_new_arange", test_ndarray_new_arange);
    CU_add_test(suite_creation, "test ndarray_new_linspace", test_ndarray_new_linspace);
    
    /* Add suite: Array Operations */
    suite_operations = CU_add_suite("Array Operations", init_suite, clean_suite);
    if (NULL == suite_operations) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(suite_operations, "test get/set 2D", test_ndarray_get_set_2d);
    CU_add_test(suite_operations, "test get/set 3D", test_ndarray_get_set_3d);
    CU_add_test(suite_operations, "test get/set 4D", test_ndarray_get_set_4d);
    CU_add_test(suite_operations, "test copy 2D", test_ndarray_copy_2d);
    CU_add_test(suite_operations, "test copy 3D", test_ndarray_copy_3d);
    
    /* Add suite: Arithmetic Operations */
    suite_arithmetic = CU_add_suite("Arithmetic Operations", init_suite, clean_suite);
    if (NULL == suite_arithmetic) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(suite_arithmetic, "test add 2D", test_ndarray_add_2d);
    CU_add_test(suite_arithmetic, "test add 3D", test_ndarray_add_3d);
    CU_add_test(suite_arithmetic, "test mul 2D", test_ndarray_mul_2d);
    CU_add_test(suite_arithmetic, "test mul 3D", test_ndarray_mul_3d);
    CU_add_test(suite_arithmetic, "test add_scalar", test_ndarray_add_scalar);
    CU_add_test(suite_arithmetic, "test mul_scalar", test_ndarray_mul_scalar);
    CU_add_test(suite_arithmetic, "test mapfnc", test_ndarray_mapfnc);
    
    /* Add suite: Matrix Multiplication */
    suite_matmul = CU_add_suite("Matrix Multiplication", init_suite, clean_suite);
    if (NULL == suite_matmul) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(suite_matmul, "test matmul 2D", test_ndarray_matmul_2d);
    CU_add_test(suite_matmul, "test matmul 3D batch", test_ndarray_matmul_3d_batch);
    CU_add_test(suite_matmul, "test matmul 4D batch", test_ndarray_matmul_4d_batch);
    
    /* Add suite: Tensordot */
    suite_tensordot = CU_add_suite("Tensor Contraction", init_suite, clean_suite);
    if (NULL == suite_tensordot) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(suite_tensordot, "test tensordot 2D matmul", test_ndarray_tensordot_2d_matmul);
    CU_add_test(suite_tensordot, "test tensordot 3D single axis", test_ndarray_tensordot_3d_single_axis);
    CU_add_test(suite_tensordot, "test tensordot multi-axis", test_ndarray_tensordot_multi_axis);
    CU_add_test(suite_tensordot, "test tensordot vs matmul", test_ndarray_tensordot_vs_matmul);
    CU_add_test(suite_tensordot, "test tensordot outer product", test_ndarray_tensordot_outer_product);
    
    /* Add suite: Stack */
    suite_stack = CU_add_suite("Stack Operations", init_suite, clean_suite);
    if (NULL == suite_stack) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(suite_stack, "test stack axis 0 2D", test_ndarray_stack_axis0_2d);
    CU_add_test(suite_stack, "test stack axis 1 2D", test_ndarray_stack_axis1_2d);
    CU_add_test(suite_stack, "test stack axis 2 2D", test_ndarray_stack_axis2_2d);
    CU_add_test(suite_stack, "test stack 3D", test_ndarray_stack_3d);
    
    /* Add suite: Concatenate */
    suite_concat = CU_add_suite("Concatenate Operations", init_suite, clean_suite);
    if (NULL == suite_concat) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(suite_concat, "test concat axis 0 2D", test_ndarray_concat_axis0_2d);
    CU_add_test(suite_concat, "test concat axis 1 2D", test_ndarray_concat_axis1_2d);
    CU_add_test(suite_concat, "test concat 3D middle", test_ndarray_concat_3d_middle);
    CU_add_test(suite_concat, "test concat multiple", test_ndarray_concat_multiple);
    
    /* Add suite: Take (Subregion) */
    suite_take = CU_add_suite("Take Subregion", init_suite, clean_suite);
    if (NULL == suite_take) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(suite_take, "test take axis 0 2D", test_ndarray_take_axis0_2d);
    CU_add_test(suite_take, "test take axis 1 2D", test_ndarray_take_axis1_2d);
    CU_add_test(suite_take, "test take 3D", test_ndarray_take_3d);
    CU_add_test(suite_take, "test take single element", test_ndarray_take_single_element);
    
    /* Add suite: Transpose */
    suite_transpose = CU_add_suite("Transpose", init_suite, clean_suite);
    if (NULL == suite_transpose) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(suite_transpose, "test transpose 2D", test_ndarray_transpose_2d);
    CU_add_test(suite_transpose, "test transpose 3D", test_ndarray_transpose_3d);
    CU_add_test(suite_transpose, "test transpose 4D", test_ndarray_transpose_4d);
    
    /* Add suite: Aggregations */
    suite_aggregation = CU_add_suite("Aggregations", init_suite, clean_suite);
    if (NULL == suite_aggregation) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    CU_add_test(suite_aggregation, "test sum all 2D", test_ndarray_aggr_sum_all_2d);
    CU_add_test(suite_aggregation, "test mean all 3D", test_ndarray_aggr_mean_all_3d);
    CU_add_test(suite_aggregation, "test max all 4D", test_ndarray_aggr_max_all_4d);
    CU_add_test(suite_aggregation, "test min all 2D", test_ndarray_aggr_min_all_2d);
    CU_add_test(suite_aggregation, "test sum axis0 2D", test_ndarray_aggr_sum_axis0_2d);
    CU_add_test(suite_aggregation, "test sum axis1 2D", test_ndarray_aggr_sum_axis1_2d);
    CU_add_test(suite_aggregation, "test mean axis0 3D", test_ndarray_aggr_mean_axis0_3d);
    CU_add_test(suite_aggregation, "test max axis2 3D", test_ndarray_aggr_max_axis2_3d);
    CU_add_test(suite_aggregation, "test min axis1 4D", test_ndarray_aggr_min_axis1_4d);
    
    /* Run all tests using the CUnit Basic interface */
    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    
    int failures = CU_get_number_of_failures();
    CU_cleanup_registry();
    
    return failures > 0 ? 1 : 0;
}
