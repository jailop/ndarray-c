#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdarg.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "ndarray.h"

#define M_PI 3.14159265358979323846

double sine_func(double x) {
    return sin(x) + 0.5 * sin(2 * x);
}

NDArray generate_serie1(size_t size) {
    NDArray ts = ndarray_new_linspace(NDA_DIMS(1, size), 0.0, 4 * M_PI, size);
    ndarray_mapfnc(ts, sine_func);
    return ts;
}

NDArray generate_serie2(size_t size) {
    size_t s_1 = size / 2, s_2 = size / 5;
    size_t s_3 = size - s_1 - s_2;
    NDArray b_1 = ndarray_new_linspace(NDA_DIMS(1, s_1), 0.0, 1.5 * M_PI, s_1);
    NDArray b_2 = ndarray_new_linspace(NDA_DIMS(1, s_2), 1.5 * M_PI, 3.0 * M_PI,
            s_2);
    NDArray b_3 = ndarray_new_linspace(NDA_DIMS(1, s_3), 3.0 * M_PI, 4.0 * M_PI,
            s_3);
    NDArray B = ndarray_new_concat(1, NDA_LIST(b_1, b_2, b_3));
    NDArray noise = ndarray_new_randnorm(NDA_DIMS(1, size), 0.0, 0.1);
    ndarray_add(B, noise);
    ndarray_mapfnc(B, sine_func);
    ndarray_free_all(NDA_LIST(b_1, b_2, b_3, noise));
    return B;
}

NDArray ts_crossprod(NDArray a, NDArray b) {
    size_t a_size = a->dims[1];
    size_t b_size = b->dims[1];
    NDArray result = ndarray_new(NDA_DIMS(a_size, b_size, 2));
    for (size_t i = 0; i < a_size; ++i) {
        double val_a = a->data[i];
        size_t offset = ndarray_offset(result, NDA_POS(i, 0, 0));
        for (size_t j = 0; j < b_size; ++j) {
            double val_b = b->data[j];
            result->data[offset++] = val_a;
            result->data[offset++] = val_b;
        }
    }
    return result;
}

int main() {
    size_t n = 100;
    NDArray A = generate_serie1(n);
    NDArray B = generate_serie2(n);
    ndarray_print(A, "Serie A", 4);
    ndarray_print(B, "Serie B", 4);
    NDArray crossprod = ts_crossprod(A, B);
    ndarray_print(crossprod, "Cross Product of A and B", 4);
    ndarray_free_all(NDA_LIST(A, B, crossprod));
    return 0;
}
