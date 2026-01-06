/**
 * test_common.c - Common utilities implementation
 */

#include "test_common.h"

int init_suite(void) {
    return 0;
}

int clean_suite(void) {
    return 0;
}

int double_equals(double a, double b) {
    return fabs(a - b) < EPSILON;
}
