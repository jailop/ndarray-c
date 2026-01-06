/**
 * test_arithmetic.c - Tests for arithmetic operations
 */

#include "test_common.h"

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
    
    ndarray_free(A);
}

void test_ndarray_axpby_basic(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 2.0);
    NDArray B = ndarray_new_full(dims, 3.0);
    
    // A = 2*A + 3*B = 2*2 + 3*3 = 4 + 9 = 13
    ndarray_axpby(A, 2.0, B, 3.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 13.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_axpby_subtract(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 10.0);
    NDArray B = ndarray_new_full(dims, 3.0);
    
    // A = 1*A - 1*B = 10 - 3 = 7
    ndarray_axpby(A, 1.0, B, -1.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 7.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_axpby_average(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 4.0);
    NDArray B = ndarray_new_full(dims, 8.0);
    
    // A = 0.5*A + 0.5*B = 0.5*4 + 0.5*8 = 2 + 4 = 6
    ndarray_axpby(A, 0.5, B, 0.5);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 6.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_axpby_3d(void) {
    size_t dims[] = {2, 2, 2, 0};
    NDArray A = ndarray_new_arange(dims, 1.0, 9.0, 1.0);
    NDArray B = ndarray_new_full(dims, 2.0);
    
    // A = 3*A + 4*B = 3*[1,2,3,4,5,6,7,8] + 4*2
    ndarray_axpby(A, 3.0, B, 4.0);
    
    // Expected: 3*1 + 8 = 11, 3*2 + 8 = 14, etc.
    CU_ASSERT_DOUBLE_EQUAL(A->data[0], 11.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[1], 14.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[2], 17.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[7], 32.0, EPSILON);
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_axpby_zero_coefficients(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 5.0);
    NDArray B = ndarray_new_full(dims, 7.0);
    
    // A = 0*A + 1*B = B
    ndarray_axpby(A, 0.0, B, 1.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 7.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_scale_shift(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 4.0);
    
    // A = 2*A + 3 = 2*4 + 3 = 11
    ndarray_scale_shift(A, 2.0, 3.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 11.0, EPSILON);
    }
    
    ndarray_free(A);
}

void test_ndarray_scale_shift_half(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_arange(dims, 2.0, 8.0, 1.0);
    
    // A = 0.5*A + 0 (halve values)
    ndarray_scale_shift(A, 0.5, 0.0);
    
    CU_ASSERT_DOUBLE_EQUAL(A->data[0], 1.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[1], 1.5, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(A->data[5], 3.5, EPSILON);
    
    ndarray_free(A);
}

void test_ndarray_mul_scaled(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 2.0);
    NDArray B = ndarray_new_full(dims, 3.0);
    
    // A = A * B * 4 = 2 * 3 * 4 = 24
    ndarray_mul_scaled(A, B, 4.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 24.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_mul_scaled_half(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_arange(dims, 1.0, 7.0, 1.0);
    NDArray B = ndarray_new_full(dims, 2.0);
    
    // A = A * B * 0.5 = [1,2,3,4,5,6] * 2 * 0.5 = [1,2,3,4,5,6]
    ndarray_mul_scaled(A, B, 0.5);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], (double)(i + 1), EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
}

double sqrt_func(double x) {
    return sqrt(x);
}

void test_ndarray_map_mul(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 4.0);
    NDArray B = ndarray_new_full(dims, 3.0);
    
    // A = sqrt(A) * B * 2 = sqrt(4) * 3 * 2 = 2 * 3 * 2 = 12
    ndarray_map_mul(A, sqrt_func, B, 2.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 12.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
}

double identity_func(double x) {
    return x;
}

void test_ndarray_map_mul_identity(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_arange(dims, 1.0, 7.0, 1.0);
    NDArray B = ndarray_new_full(dims, 2.0);
    
    // With identity function: A = A * B * 1 = [1,2,3,4,5,6] * 2
    ndarray_map_mul(A, identity_func, B, 1.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], (double)(i + 1) * 2.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
}

void test_ndarray_mul_add_basic(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 2.0);
    NDArray B = ndarray_new_full(dims, 3.0);
    NDArray C = ndarray_new_full(dims, 5.0);
    
    // C = 2 * (A * B) + 0.5 * C = 2 * (2*3) + 0.5 * 5 = 12 + 2.5 = 14.5
    ndarray_mul_add(A, B, C, 2.0, 0.5);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 14.5, EPSILON);
    }
    
    ndarray_free_all(NDA_LIST(A, B, C));
}

void test_ndarray_mul_add_accumulate(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 2.0);
    NDArray B = ndarray_new_full(dims, 3.0);
    NDArray C = ndarray_new_full(dims, 10.0);
    
    // C = 1 * (A * B) + 1 * C = 6 + 10 = 16
    ndarray_mul_add(A, B, C, 1.0, 1.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 16.0, EPSILON);
    }
    
    ndarray_free_all(NDA_LIST(A, B, C));
}

void test_ndarray_mul_add_replace(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 4.0);
    NDArray B = ndarray_new_full(dims, 5.0);
    NDArray C = ndarray_new_full(dims, 100.0);
    
    // C = 1 * (A * B) + 0 * C = 20
    ndarray_mul_add(A, B, C, 1.0, 0.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(C->data[i], 20.0, EPSILON);
    }
    
    ndarray_free_all(NDA_LIST(A, B, C));
}

void test_ndarray_gemv_basic(void) {
    // A: 3x4 matrix, x: 4x1 vector, y: 3x1 vector
    size_t dims_a[] = {3, 4, 0};
    size_t dims_x[] = {4, 1, 0};
    size_t dims_y[] = {3, 1, 0};
    
    NDArray A = ndarray_new_ones(dims_a);
    NDArray x = ndarray_new_full(dims_x, 2.0);
    NDArray y = ndarray_new_zeros(dims_y);
    
    // y = 1.0 * A * x + 0.0 * y = A * [2,2,2,2]^T = [8,8,8]^T
    ndarray_gemv(A, x, 1.0, 0.0, y);
    
    for (size_t i = 0; i < 3; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(y->data[i], 8.0, EPSILON);
    }
    
    ndarray_free_all(NDA_LIST(A, x, y));
}

void test_ndarray_gemv_accumulate(void) {
    // A: 2x3 matrix, x: 3x1 vector, y: 2x1 vector
    size_t dims_a[] = {2, 3, 0};
    size_t dims_x[] = {3, 1, 0};
    size_t dims_y[] = {2, 1, 0};
    
    NDArray A = ndarray_new_ones(dims_a);
    NDArray x = ndarray_new_full(dims_x, 1.0);
    NDArray y = ndarray_new_full(dims_y, 5.0);
    
    // y = 1.0 * A * x + 1.0 * y = [3,3]^T + [5,5]^T = [8,8]^T
    ndarray_gemv(A, x, 1.0, 1.0, y);
    
    CU_ASSERT_DOUBLE_EQUAL(y->data[0], 8.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(y->data[1], 8.0, EPSILON);
    
    ndarray_free_all(NDA_LIST(A, x, y));
}

void test_ndarray_gemv_scaled(void) {
    // Test with alpha and beta != 1
    size_t dims_a[] = {2, 2, 0};
    size_t dims_x[] = {2, 1, 0};
    size_t dims_y[] = {2, 1, 0};
    
    NDArray A = ndarray_new(dims_a);
    A->data[0] = 1.0; A->data[1] = 2.0;
    A->data[2] = 3.0; A->data[3] = 4.0;
    
    NDArray x = ndarray_new(dims_x);
    x->data[0] = 1.0; x->data[1] = 1.0;
    
    NDArray y = ndarray_new_full(dims_y, 10.0);
    
    // y = 2.0 * A * x + 0.5 * y
    // A*x = [3, 7]^T, result = 2*[3,7] + 0.5*[10,10] = [6,14] + [5,5] = [11,19]
    ndarray_gemv(A, x, 2.0, 0.5, y);
    
    CU_ASSERT_DOUBLE_EQUAL(y->data[0], 11.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(y->data[1], 19.0, EPSILON);
    
    ndarray_free_all(NDA_LIST(A, x, y));
}

/* Register tests for this module */
void register_arithmetic_tests(CU_pSuite suite) {
    CU_add_test(suite, "test add 2d", test_ndarray_add_2d);
    CU_add_test(suite, "test add 3d", test_ndarray_add_3d);
    CU_add_test(suite, "test mul 2d", test_ndarray_mul_2d);
    CU_add_test(suite, "test mul 3d", test_ndarray_mul_3d);
    CU_add_test(suite, "test add scalar", test_ndarray_add_scalar);
    CU_add_test(suite, "test mul scalar", test_ndarray_mul_scalar);
    CU_add_test(suite, "test mapfnc", test_ndarray_mapfnc);
    CU_add_test(suite, "test axpby basic", test_ndarray_axpby_basic);
    CU_add_test(suite, "test axpby subtract", test_ndarray_axpby_subtract);
    CU_add_test(suite, "test axpby average", test_ndarray_axpby_average);
    CU_add_test(suite, "test axpby 3d", test_ndarray_axpby_3d);
    CU_add_test(suite, "test axpby zero coefficients", test_ndarray_axpby_zero_coefficients);
    CU_add_test(suite, "test scale shift", test_ndarray_scale_shift);
    CU_add_test(suite, "test scale shift half", test_ndarray_scale_shift_half);
    CU_add_test(suite, "test mul scaled", test_ndarray_mul_scaled);
    CU_add_test(suite, "test mul scaled half", test_ndarray_mul_scaled_half);
    CU_add_test(suite, "test map mul", test_ndarray_map_mul);
    CU_add_test(suite, "test map mul identity", test_ndarray_map_mul_identity);
    CU_add_test(suite, "test mul add basic", test_ndarray_mul_add_basic);
    CU_add_test(suite, "test mul add accumulate", test_ndarray_mul_add_accumulate);
    CU_add_test(suite, "test mul add replace", test_ndarray_mul_add_replace);
    CU_add_test(suite, "test gemv basic", test_ndarray_gemv_basic);
    CU_add_test(suite, "test gemv accumulate", test_ndarray_gemv_accumulate);
    CU_add_test(suite, "test gemv scaled", test_ndarray_gemv_scaled);
}
