/**
 * test_main.c - Main test runner for ndarray library
 * 
 * Modular test suite with tests organized into separate files
 */

#include "test_common.h"
#include "test_modules.h"

int main() {
    /* Initialize CUnit registry */
    if (CUE_SUCCESS != CU_initialize_registry()) {
        return CU_get_error();
    }

    /* Create test suites */
    CU_pSuite suite_creation = CU_add_suite("Array Creation", init_suite, clean_suite);
    CU_pSuite suite_operations = CU_add_suite("Array Operations", init_suite, clean_suite);
    CU_pSuite suite_arithmetic = CU_add_suite("Arithmetic Operations", init_suite, clean_suite);
    CU_pSuite suite_matmul = CU_add_suite("Matrix Multiplication", init_suite, clean_suite);
    CU_pSuite suite_tensordot = CU_add_suite("Tensor Contraction", init_suite, clean_suite);
    CU_pSuite suite_stack = CU_add_suite("Stack Operations", init_suite, clean_suite);
    CU_pSuite suite_concat = CU_add_suite("Concatenate Operations", init_suite, clean_suite);
    CU_pSuite suite_take = CU_add_suite("Take Subregion", init_suite, clean_suite);
    CU_pSuite suite_transpose = CU_add_suite("Transpose", init_suite, clean_suite);
    CU_pSuite suite_reshape = CU_add_suite("Reshape", init_suite, clean_suite);
    CU_pSuite suite_aggregation = CU_add_suite("Aggregations", init_suite, clean_suite);
    CU_pSuite suite_conditional = CU_add_suite("Conditional Operations", init_suite, clean_suite);
    CU_pSuite suite_slice = CU_add_suite("Slice Access", init_suite, clean_suite);
    CU_pSuite suite_comparison = CU_add_suite("Comparison Operations", init_suite, clean_suite);
    CU_pSuite suite_randquality = CU_add_suite("Random Number Quality", init_suite, clean_suite);

    /* Check suite creation */
    if (!suite_creation || !suite_operations || !suite_arithmetic || 
        !suite_matmul || !suite_tensordot || !suite_stack || !suite_concat ||
        !suite_take || !suite_transpose || !suite_reshape || !suite_aggregation ||
        !suite_conditional || !suite_slice || !suite_comparison || !suite_randquality) {
        CU_cleanup_registry();
        return CU_get_error();
    }

    /* Register tests from each module */
    register_creation_tests(suite_creation);
    register_operations_tests(suite_operations);
    register_arithmetic_tests(suite_arithmetic);
    register_matmul_tests(suite_matmul);
    register_tensordot_tests(suite_tensordot);
    register_stack_tests(suite_stack);
    register_concat_tests(suite_concat);
    register_take_tests(suite_take);
    register_transpose_tests(suite_transpose);
    register_reshape_tests(suite_reshape);
    register_aggregation_tests(suite_aggregation);
    register_conditional_tests(suite_conditional);
    register_slice_tests(suite_slice);
    register_comparison_tests(suite_comparison);
    register_randquality_tests(suite_randquality);

    /* Run all tests */
    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();

    int failures = CU_get_number_of_failures();
    CU_cleanup_registry();

    return failures > 0 ? 1 : 0;
}
