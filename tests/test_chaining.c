/**
 * test_chaining.c - Tests for chained/nested function calls
 * 
 * Tests that in-place operations can be chained together since they
 * return pointers to the modified arrays.
 */

#include "test_common.h"
#include <math.h>

/* Helper functions for testing mapfnc */
static double square_fn(double x) {
    return x * x;
}

static double sqrt_fn(double x) {
    return sqrt(x);
}

/* Test basic two-operation chain */
void test_chain_add_mul_scalar(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_ones(dims);
    
    // Chain: (A + 5) * 2 = (1 + 5) * 2 = 12
    ndarray_mul_scalar(ndarray_add_scalar(A, 5.0), 2.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 12.0, EPSILON);
    }
    
    ndarray_free(A);
}

/* Test three-operation chain */
void test_chain_mul_add_clip(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 3.0);
    
    // Chain: clip((A * 2) + 1, 0, 5) = clip((3*2) + 1, 0, 5) = clip(7, 0, 5) = 5
    ndarray_clip(
        ndarray_add_scalar(
            ndarray_mul_scalar(A, 2.0),
            1.0
        ),
        0.0, 5.0
    );
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 5.0, EPSILON);
    }
    
    ndarray_free(A);
}

/* Test chaining with abs */
void test_chain_abs_mul_scalar(void) {
    size_t dims[] = {1, 5, 0};
    double data[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    NDArray A = ndarray_new_from_data(dims, data);
    
    // Chain: abs(A) * 2
    ndarray_mul_scalar(ndarray_abs(A), 2.0);
    
    double expected[] = {4.0, 2.0, 0.0, 2.0, 4.0};
    for (size_t i = 0; i < 5; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], expected[i], EPSILON);
    }
    
    ndarray_free(A);
}

/* Test chaining with clip_min (ReLU pattern) */
void test_chain_relu_scale(void) {
    size_t dims[] = {1, 5, 0};
    double data[] = {-2.0, -1.0, 0.0, 1.0, 2.0};
    NDArray A = ndarray_new_from_data(dims, data);
    
    // Chain: max(0, A) * 0.5  (ReLU then scale)
    ndarray_mul_scalar(ndarray_clip_min(A, 0.0), 0.5);
    
    double expected[] = {0.0, 0.0, 0.0, 0.5, 1.0};
    for (size_t i = 0; i < 5; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], expected[i], EPSILON);
    }
    
    ndarray_free(A);
}

/* Test chaining with clip_max */
void test_chain_clip_max_add(void) {
    size_t dims[] = {1, 5, 0};
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    NDArray A = ndarray_new_from_data(dims, data);
    
    // Chain: clip_max(A, 3) + 10
    ndarray_add_scalar(ndarray_clip_max(A, 3.0), 10.0);
    
    double expected[] = {11.0, 12.0, 13.0, 13.0, 13.0};
    for (size_t i = 0; i < 5; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], expected[i], EPSILON);
    }
    
    ndarray_free(A);
}

/* Test chaining with clip (both min and max) */
void test_chain_clip_range_mul(void) {
    size_t dims[] = {1, 5, 0};
    double data[] = {-2.0, -1.0, 0.0, 5.0, 10.0};
    NDArray A = ndarray_new_from_data(dims, data);
    
    // Chain: clip(A, 0, 5) * 2
    ndarray_mul_scalar(ndarray_clip(A, 0.0, 5.0), 2.0);
    
    double expected[] = {0.0, 0.0, 0.0, 10.0, 10.0};
    for (size_t i = 0; i < 5; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], expected[i], EPSILON);
    }
    
    ndarray_free(A);
}

/* Test chaining with mapfnc */
void test_chain_mapfnc_add_scalar(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 2.0);
    
    // Chain: square(A) + 1 = 2^2 + 1 = 5
    ndarray_add_scalar(ndarray_mapfnc(A, square_fn), 1.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 5.0, EPSILON);
    }
    
    ndarray_free(A);
}

/* Test complex chain: abs, add, sqrt, clip */
void test_chain_complex_transform(void) {
    size_t dims[] = {1, 5, 0};
    double data[] = {-10.0, -5.0, 0.0, 5.0, 10.0};
    NDArray A = ndarray_new_from_data(dims, data);
    
    // Chain: clip_max(sqrt(abs(A) + 1), 3.0)
    ndarray_clip_max(
        ndarray_mapfnc(
            ndarray_add_scalar(
                ndarray_abs(A),
                1.0
            ),
            sqrt_fn
        ),
        3.0
    );
    
    // Expected: sqrt(abs(x) + 1) clipped to max 3
    // -10 -> abs(10) + 1 = 11 -> sqrt(11) = 3.316... -> 3.0
    // -5  -> abs(5) + 1 = 6 -> sqrt(6) = 2.449...
    // 0   -> abs(0) + 1 = 1 -> sqrt(1) = 1.0
    // 5   -> abs(5) + 1 = 6 -> sqrt(6) = 2.449...
    // 10  -> abs(10) + 1 = 11 -> sqrt(11) = 3.316... -> 3.0
    double expected[] = {3.0, sqrt(6.0), 1.0, sqrt(6.0), 3.0};
    
    for (size_t i = 0; i < 5; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], expected[i], EPSILON);
    }
    
    ndarray_free(A);
}

/* Test chaining with element-wise operations */
void test_chain_add_mul_arrays(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_full(dims, 2.0);
    NDArray B = ndarray_new_full(dims, 3.0);
    NDArray C = ndarray_new_full(dims, 5.0);
    
    // Chain: (A + B) * scalar = (2 + 3) * 2 = 10
    // Note: Can't chain element-wise mul with another array, so use scalar
    ndarray_mul_scalar(ndarray_add(A, B), 2.0);
    
    for (size_t i = 0; i < 6; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 10.0, EPSILON);
    }
    
    ndarray_free(A);
    ndarray_free(B);
    ndarray_free(C);
}

/* Test chaining with sign function */
void test_chain_sign_mul(void) {
    size_t dims[] = {1, 5, 0};
    double data[] = {-3.0, -1.0, 0.0, 2.0, 5.0};
    NDArray A = ndarray_new_from_data(dims, data);
    
    // Chain: sign(A) * 10
    ndarray_mul_scalar(ndarray_sign(A), 10.0);
    
    double expected[] = {-10.0, -10.0, 0.0, 10.0, 10.0};
    for (size_t i = 0; i < 5; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], expected[i], EPSILON);
    }
    
    ndarray_free(A);
}

/* Test normalized values through clipping and scaling */
void test_chain_normalize_pattern(void) {
    size_t dims[] = {1, 5, 0};
    double data[] = {-10.0, -5.0, 0.0, 5.0, 10.0};
    NDArray A = ndarray_new_from_data(dims, data);
    
    // Chain: normalize to [0,1] range
    // Step 1: clip to [0, 10]
    // Step 2: divide by 10 (multiply by 0.1)
    ndarray_mul_scalar(ndarray_clip(A, 0.0, 10.0), 0.1);
    
    double expected[] = {0.0, 0.0, 0.0, 0.5, 1.0};
    for (size_t i = 0; i < 5; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], expected[i], EPSILON);
    }
    
    ndarray_free(A);
}

/* Test multi-dimensional array chaining */
void test_chain_3d_arrays(void) {
    size_t dims[] = {2, 2, 2, 0};
    NDArray A = ndarray_new_full(dims, 3.0);
    
    // Chain: ((A * 2) + 4) clipped to [0, 15]
    // 3 * 2 = 6, 6 + 4 = 10, clip(10, 0, 15) = 10
    ndarray_clip(
        ndarray_add_scalar(
            ndarray_mul_scalar(A, 2.0),
            4.0
        ),
        0.0, 15.0
    );
    
    for (size_t i = 0; i < 8; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 10.0, EPSILON);
    }
    
    ndarray_free(A);
}

/* Test chaining preserves array structure */
void test_chain_preserves_shape(void) {
    size_t dims[] = {3, 4, 0};
    NDArray A = ndarray_new_ones(dims);
    
    // Apply several chained operations
    ndarray_mul_scalar(
        ndarray_add_scalar(
            ndarray_mul_scalar(A, 2.0),
            3.0
        ),
        0.5
    );
    
    // Verify shape is preserved
    CU_ASSERT_EQUAL(A->ndim, 2);
    CU_ASSERT_EQUAL(A->dims[0], 3);
    CU_ASSERT_EQUAL(A->dims[1], 4);
    
    // Verify values: ((1 * 2) + 3) * 0.5 = 5 * 0.5 = 2.5
    for (size_t i = 0; i < 12; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(A->data[i], 2.5, EPSILON);
    }
    
    ndarray_free(A);
}

/* Register all chaining tests */
void register_chaining_tests(CU_pSuite suite) {
    CU_add_test(suite, "Chain: add and mul scalar", test_chain_add_mul_scalar);
    CU_add_test(suite, "Chain: mul, add, clip", test_chain_mul_add_clip);
    CU_add_test(suite, "Chain: abs and mul", test_chain_abs_mul_scalar);
    CU_add_test(suite, "Chain: ReLU and scale", test_chain_relu_scale);
    CU_add_test(suite, "Chain: clip_max and add", test_chain_clip_max_add);
    CU_add_test(suite, "Chain: clip range and mul", test_chain_clip_range_mul);
    CU_add_test(suite, "Chain: mapfnc and add", test_chain_mapfnc_add_scalar);
    CU_add_test(suite, "Chain: complex transform", test_chain_complex_transform);
    CU_add_test(suite, "Chain: element-wise ops", test_chain_add_mul_arrays);
    CU_add_test(suite, "Chain: sign and mul", test_chain_sign_mul);
    CU_add_test(suite, "Chain: normalize pattern", test_chain_normalize_pattern);
    CU_add_test(suite, "Chain: 3D arrays", test_chain_3d_arrays);
    CU_add_test(suite, "Chain: preserves shape", test_chain_preserves_shape);
}
