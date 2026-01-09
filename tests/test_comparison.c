/**
 * test_comparison.c - Tests for comparison operations
 */

#include "test_common.h"

void test_ndarray_equal(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new(dims);
    NDArray B = ndarray_new(dims);
    
    A->data[0] = 1.0; B->data[0] = 1.0;
    A->data[1] = 2.0; B->data[1] = 3.0;
    A->data[2] = 3.0; B->data[2] = 3.0;
    A->data[3] = 4.0; B->data[3] = 4.0;
    A->data[4] = 5.0; B->data[4] = 6.0;
    A->data[5] = 7.0; B->data[5] = 7.0;
    
    NDArray result = ndarray_new_equal(A, B);
    
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 1.0, EPSILON);  // equal
    CU_ASSERT_DOUBLE_EQUAL(result->data[1], 0.0, EPSILON);  // not equal
    CU_ASSERT_DOUBLE_EQUAL(result->data[2], 1.0, EPSILON);  // equal
    CU_ASSERT_DOUBLE_EQUAL(result->data[3], 1.0, EPSILON);  // equal
    CU_ASSERT_DOUBLE_EQUAL(result->data[4], 0.0, EPSILON);  // not equal
    CU_ASSERT_DOUBLE_EQUAL(result->data[5], 1.0, EPSILON);  // equal
    
    ndarray_free_all(NDA_LIST(A, B, result));
}

void test_ndarray_less_greater(void) {
    size_t dims[] = {2, 2, 0};
    NDArray A = ndarray_new_arange(dims, 1.0, 5.0, 1.0);
    NDArray B = ndarray_new_full(dims, 2.5);
    
    NDArray less = ndarray_new_less(A, B);
    CU_ASSERT_DOUBLE_EQUAL(less->data[0], 1.0, EPSILON);  // 1 < 2.5
    CU_ASSERT_DOUBLE_EQUAL(less->data[1], 1.0, EPSILON);  // 2 < 2.5
    CU_ASSERT_DOUBLE_EQUAL(less->data[2], 0.0, EPSILON);  // 3 < 2.5
    CU_ASSERT_DOUBLE_EQUAL(less->data[3], 0.0, EPSILON);  // 4 < 2.5
    
    NDArray greater = ndarray_new_greater(A, B);
    CU_ASSERT_DOUBLE_EQUAL(greater->data[0], 0.0, EPSILON);  // 1 > 2.5
    CU_ASSERT_DOUBLE_EQUAL(greater->data[1], 0.0, EPSILON);  // 2 > 2.5
    CU_ASSERT_DOUBLE_EQUAL(greater->data[2], 1.0, EPSILON);  // 3 > 2.5
    CU_ASSERT_DOUBLE_EQUAL(greater->data[3], 1.0, EPSILON);  // 4 > 2.5
    
    ndarray_free_all(NDA_LIST(A, B, less, greater));
}

void test_ndarray_scalar_comparison(void) {
    size_t dims[] = {2, 3, 0};
    NDArray A = ndarray_new_arange(dims, 0.0, 6.0, 1.0);
    
    NDArray eq = ndarray_new_equal_scalar(A, 3.0);
    CU_ASSERT_DOUBLE_EQUAL(eq->data[3], 1.0, EPSILON);  // Only index 3 == 3.0
    CU_ASSERT_DOUBLE_EQUAL(eq->data[0], 0.0, EPSILON);
    CU_ASSERT_DOUBLE_EQUAL(eq->data[5], 0.0, EPSILON);
    
    NDArray lt = ndarray_new_less_scalar(A, 3.0);
    CU_ASSERT_DOUBLE_EQUAL(lt->data[0], 1.0, EPSILON);  // 0 < 3
    CU_ASSERT_DOUBLE_EQUAL(lt->data[1], 1.0, EPSILON);  // 1 < 3
    CU_ASSERT_DOUBLE_EQUAL(lt->data[2], 1.0, EPSILON);  // 2 < 3
    CU_ASSERT_DOUBLE_EQUAL(lt->data[3], 0.0, EPSILON);  // 3 < 3
    
    NDArray gt = ndarray_new_greater_scalar(A, 3.0);
    CU_ASSERT_DOUBLE_EQUAL(gt->data[4], 1.0, EPSILON);  // 4 > 3
    CU_ASSERT_DOUBLE_EQUAL(gt->data[5], 1.0, EPSILON);  // 5 > 3
    CU_ASSERT_DOUBLE_EQUAL(gt->data[2], 0.0, EPSILON);  // 2 > 3
    
    ndarray_free_all(NDA_LIST(A, eq, lt, gt));
}

void test_ndarray_new_logical_and_or(void) {
    size_t dims[] = {2, 2, 0};
    NDArray A = ndarray_new(dims);
    NDArray B = ndarray_new(dims);
    
    A->data[0] = 1.0; B->data[0] = 1.0;
    A->data[1] = 1.0; B->data[1] = 0.0;
    A->data[2] = 0.0; B->data[2] = 1.0;
    A->data[3] = 0.0; B->data[3] = 0.0;
    
    NDArray and_result = ndarray_new_logical_and(A, B);
    CU_ASSERT_DOUBLE_EQUAL(and_result->data[0], 1.0, EPSILON);  // T && T
    CU_ASSERT_DOUBLE_EQUAL(and_result->data[1], 0.0, EPSILON);  // T && F
    CU_ASSERT_DOUBLE_EQUAL(and_result->data[2], 0.0, EPSILON);  // F && T
    CU_ASSERT_DOUBLE_EQUAL(and_result->data[3], 0.0, EPSILON);  // F && F
    
    NDArray or_result = ndarray_new_logical_or(A, B);
    CU_ASSERT_DOUBLE_EQUAL(or_result->data[0], 1.0, EPSILON);  // T || T
    CU_ASSERT_DOUBLE_EQUAL(or_result->data[1], 1.0, EPSILON);  // T || F
    CU_ASSERT_DOUBLE_EQUAL(or_result->data[2], 1.0, EPSILON);  // F || T
    CU_ASSERT_DOUBLE_EQUAL(or_result->data[3], 0.0, EPSILON);  // F || F
    
    ndarray_free_all(NDA_LIST(A, B, and_result, or_result));
}

void test_ndarray_new_logical_not(void) {
    size_t dims[] = {2, 2, 0};
    NDArray A = ndarray_new(dims);
    
    A->data[0] = 1.0;
    A->data[1] = 0.0;
    A->data[2] = 5.0;
    A->data[3] = 0.0;
    
    NDArray result = ndarray_new_logical_not(A);
    
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 0.0, EPSILON);  // !T
    CU_ASSERT_DOUBLE_EQUAL(result->data[1], 1.0, EPSILON);  // !F
    CU_ASSERT_DOUBLE_EQUAL(result->data[2], 0.0, EPSILON);  // !T
    CU_ASSERT_DOUBLE_EQUAL(result->data[3], 1.0, EPSILON);  // !F
    
    ndarray_free_all(NDA_LIST(A, result));
}

void test_ndarray_new_where(void) {
    size_t dims[] = {2, 3, 0};
    NDArray condition = ndarray_new(dims);
    NDArray x = ndarray_new_full(dims, 10.0);
    NDArray y = ndarray_new_full(dims, 20.0);
    
    condition->data[0] = 1.0;  // true
    condition->data[1] = 0.0;  // false
    condition->data[2] = 1.0;  // true
    condition->data[3] = 0.0;  // false
    condition->data[4] = 1.0;  // true
    condition->data[5] = 0.0;  // false
    
    NDArray result = ndarray_new_where(condition, x, y);
    
    CU_ASSERT_DOUBLE_EQUAL(result->data[0], 10.0, EPSILON);  // from x
    CU_ASSERT_DOUBLE_EQUAL(result->data[1], 20.0, EPSILON);  // from y
    CU_ASSERT_DOUBLE_EQUAL(result->data[2], 10.0, EPSILON);  // from x
    CU_ASSERT_DOUBLE_EQUAL(result->data[3], 20.0, EPSILON);  // from y
    CU_ASSERT_DOUBLE_EQUAL(result->data[4], 10.0, EPSILON);  // from x
    CU_ASSERT_DOUBLE_EQUAL(result->data[5], 20.0, EPSILON);  // from y
    
    ndarray_free_all(NDA_LIST(condition, x, y, result));
}

/* Register tests for this module */
void register_comparison_tests(CU_pSuite suite) {
    CU_add_test(suite, "test equal", test_ndarray_equal);
    CU_add_test(suite, "test less greater", test_ndarray_less_greater);
    CU_add_test(suite, "test scalar comparison", test_ndarray_scalar_comparison);
    CU_add_test(suite, "test logical and or", test_ndarray_new_logical_and_or);
    CU_add_test(suite, "test logical not", test_ndarray_new_logical_not);
    CU_add_test(suite, "test where", test_ndarray_new_where);
}
