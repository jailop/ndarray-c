/**
 * test_modules.h - Header for all test modules
 */

#ifndef TEST_MODULES_H
#define TEST_MODULES_H

#include "test_common.h"

/* Registration functions for each test module */
void register_creation_tests(CU_pSuite suite);
void register_operations_tests(CU_pSuite suite);
void register_arithmetic_tests(CU_pSuite suite);
void register_matmul_tests(CU_pSuite suite);
void register_tensordot_tests(CU_pSuite suite);
void register_stack_tests(CU_pSuite suite);
void register_concat_tests(CU_pSuite suite);
void register_take_tests(CU_pSuite suite);
void register_transpose_tests(CU_pSuite suite);
void register_reshape_tests(CU_pSuite suite);
void register_aggregation_tests(CU_pSuite suite);
void register_conditional_tests(CU_pSuite suite);
void register_slice_tests(CU_pSuite suite);
void register_comparison_tests(CU_pSuite suite);

#endif /* TEST_MODULES_H */
