/**
 * test_common.h - Common utilities for ndarray tests
 */

#ifndef TEST_COMMON_H
#define TEST_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CUnit/Basic.h>
#include "ndarray.h"

#define EPSILON 1e-6

/* Test Suite Initialization */
int init_suite(void);
int clean_suite(void);

/* Helper function to compare doubles */
int double_equals(double a, double b);

#endif /* TEST_COMMON_H */
