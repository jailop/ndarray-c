#include "ndarray_internal.h"

static void print_recursive_helper(NDArray arr, size_t *indices, int precision,
                                    size_t depth, size_t max_items_per_dim) {
    size_t dim_size = arr->dims[depth];
    int should_truncate = (dim_size > 2 * max_items_per_dim);
    if (depth == arr->ndim - 1) {
        // Print innermost dimension
        printf("[");
        for (size_t i = 0; i < dim_size; ++i) {
            if (should_truncate && i == max_items_per_dim) {
                printf("...");
                i = dim_size - max_items_per_dim - 1;
                continue;
            }
            indices[depth] = i;
            printf("%.*f", precision, ndarray_get(arr, indices));
            if (i < dim_size - 1 && 
                !(should_truncate && i == dim_size - max_items_per_dim - 1)) {
                printf(", ");
            }
        }
        printf("]");
    } else {
        printf("[");
        for (size_t i = 0; i < dim_size; ++i) {
            if (should_truncate && i == max_items_per_dim) {
                if (i > 0) {
                    printf("\n");
                    for (size_t d = 0; d <= depth; ++d) printf(" ");
                }
                printf("...");
                i = dim_size - max_items_per_dim - 1;
                continue;
            }
            
            indices[depth] = i;
            if (i > 0) {
                printf("\n");
                for (size_t d = 0; d <= depth; ++d) printf(" ");
            }
            print_recursive_helper(arr, indices, precision, depth + 1,
                    max_items_per_dim);
            if (i < dim_size - 1
                    && !(should_truncate
                    && i == dim_size - max_items_per_dim - 1)) {
                printf(",");
            }
        }
        printf("]");
    }
}

void ndarray_print(NDArray arr, const char *name, int precision) {
    assert(arr != NULL && "ndarray cannot be NULL");
    if (precision < 0) precision = 4;
    int term_width = 80; // Default fallback
#ifndef _WIN32
    struct winsize w;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) != -1 && w.ws_col > 0) {
        term_width = w.ws_col;
    }
#else
    #ifdef _MSC_VER
        #include <windows.h>
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
            term_width = csbi.srWindow.Right - csbi.srWindow.Left + 1;
        }
    #endif
#endif
    if (name != NULL) {
        printf("Array '%s' [", name);
    } else {
        printf("Array [");
    }
    for (size_t i = 0; i < arr->ndim; ++i) {
        printf("%zu%s", arr->dims[i], i < arr->ndim - 1 ? ", " : "");
    }
    printf("]:\n");
    if (arr->ndim == 2) {
        // 2D: Pretty matrix format with smart truncation
        size_t rows = arr->dims[0];
        size_t cols = arr->dims[1];
        int elem_width = precision + 6;
        int available_width = term_width - 6; // Account for brackets and spaces
        size_t max_cols = available_width / (elem_width + 1);
        // Decide how many rows/cols to show
        size_t show_rows_head = 3, show_rows_tail = 3;
        size_t show_cols_head = 3, show_cols_tail = 3;
        int truncate_rows = (rows > show_rows_head + show_rows_tail + 1);
        int truncate_cols = (cols > max_cols);
        if (truncate_cols && max_cols > 6) {
            show_cols_head = max_cols / 2;
            show_cols_tail = max_cols - show_cols_head;
        } else if (!truncate_cols) {
            show_cols_head = cols;
            show_cols_tail = 0;
        }
        printf("[");
        for (size_t i = 0; i < rows; ++i) {
            if (truncate_rows && i == show_rows_head) {
                if (i > 0) printf(" ");
                printf("\n...\n");
                i = rows - show_rows_tail - 1;
                continue;
            }
            
            if (i > 0) printf(" ");
            printf("[");
            
            for (size_t j = 0; j < cols; ++j) {
                if (truncate_cols && j == show_cols_head) {
                    printf("  ...");
                    j = cols - show_cols_tail - 1;
                    continue;
                }
                
                size_t pos[] = {i, j};
                printf("%*.*f", elem_width, precision, ndarray_get(arr, pos));
                if (j < cols - 1 && !(truncate_cols && j == cols - show_cols_tail - 1)) {
                    printf(" ");
                }
            }
            printf("]");
            if (i < rows - 1 && !(truncate_rows && i == show_rows_head - 1)) {
                printf("\n");
            }
        }
        printf("]\n");
    } else {
        // 3D+: Nested bracket notation with truncation
        size_t indices[arr->ndim];
        for (size_t i = 0; i < arr->ndim; ++i) {
            indices[i] = 0;
        }
        size_t max_items_per_dim = 3;
        
        print_recursive_helper(arr, indices, precision, 0, max_items_per_dim);
        printf("\n");
    }
}

void ndarray_print_complex(const NDArray arr, const char *name, int precision, NDAPrintMode mode) {
    assert(arr != NULL && "ndarray cannot be NULL");
    if (precision < 0) precision = 4;
    
    if (name != NULL) {
        printf("Array '%s' [", name);
    } else {
        printf("Array [");
    }
    for (size_t i = 0; i < arr->ndim; ++i) {
        printf("%zu%s", arr->dims[i], i < arr->ndim - 1 ? ", " : "");
    }
    printf("]:\n");
    
    size_t size = ndarray_size(arr);
    if (arr->dtype == NDA_COMPLEX64 || arr->dtype == NDA_COMPLEX32) {
        double complex *cdata;
        float complex *fdata;
        
        if (arr->dtype == NDA_COMPLEX64) {
            cdata = (double complex*)arr->data;
        } else {
            fdata = (float complex*)arr->data;
        }
        
        if (arr->ndim == 2 && size <= 24) {
            // Small 2D complex array - show as matrix
            int elem_width = precision + 8;
            for (size_t i = 0; i < arr->dims[0]; ++i) {
                printf("  [");
                for (size_t j = 0; j < arr->dims[1]; ++j) {
                    if (j > 0) printf(", ");
                    
                    size_t idx = i * arr->dims[1] + j;
                    double real_val, imag_val, mag, phase;
                    
                    if (arr->dtype == NDA_COMPLEX64) {
                        real_val = creal(cdata[idx]);
                        imag_val = cimag(cdata[idx]);
                        mag = cabs(cdata[idx]);
                        phase = carg(cdata[idx]);
                    } else {
                        real_val = (double)crealf(fdata[idx]);
                        imag_val = (double)cimagf(fdata[idx]);
                        mag = (double)cabsf(fdata[idx]);
                        phase = (double)cargf(fdata[idx]);
                    }
                    
                    switch (mode) {
                        case NDA_PRINT_DEFAULT:
                            printf("%*.*f%+*.*fi", elem_width/2-1, precision, real_val, 
                                   elem_width/2-1, precision, imag_val);
                            break;
                        case NDA_PRINT_MAGNITUDE:
                            printf("%*.*f", elem_width, precision, mag);
                            break;
                        case NDA_PRINT_PHASE:
                            printf("%*.*f", elem_width, precision, phase);
                            break;
                        case NDA_PRINT_REAL:
                            printf("%*.*f", elem_width, precision, real_val);
                            break;
                        case NDA_PRINT_IMAG:
                            printf("%*.*f", elem_width, precision, imag_val);
                            break;
                        case NDA_PRINT_POLAR:
                            printf("%*.*f∠%*.*f", elem_width/2-1, precision, mag,
                                   elem_width/2-1, precision, phase);
                            break;
                        default:
                            printf("%*.*f%+*.*fi", elem_width/2-1, precision, real_val,
                                   elem_width/2-1, precision, imag_val);
                            break;
                    }
                }
                printf("  ]\n");
            }
        } else {
            // Large or high-dimensional array - use truncation
            size_t *indices = (size_t*)malloc(sizeof(size_t) * arr->ndim);
            for (size_t i = 0; i < arr->ndim; ++i) {
                indices[i] = 0;
            }
            
            for (size_t i = 0; i < size; ++i) {
                double real_val, imag_val;
                
                if (arr->dtype == NDA_COMPLEX64) {
                    double complex *cdata = (double complex*)arr->data;
                    real_val = creal(cdata[i]);
                    imag_val = cimag(cdata[i]);
                } else {
                    float complex *fdata = (float complex*)arr->data;
                    real_val = (double)crealf(fdata[i]);
                    imag_val = (double)cimagf(fdata[i]);
                }
                
                switch (mode) {
                    case NDA_PRINT_DEFAULT:
                        printf("%.*f%+.*fi", precision, real_val, precision, imag_val);
                        break;
                    case NDA_PRINT_MAGNITUDE:
                        printf("%.*f", precision, sqrt(real_val*real_val + imag_val*imag_val));
                        break;
                    case NDA_PRINT_PHASE:
                        printf("%.*f", precision, atan2(imag_val, real_val));
                        break;
                    case NDA_PRINT_REAL:
                        printf("%.*f", precision, real_val);
                        break;
                    case NDA_PRINT_IMAG:
                        printf("%.*f", precision, imag_val);
                        break;
                    case NDA_PRINT_POLAR:
                        printf("%.*f∠%.*f", precision, 
                               sqrt(real_val*real_val + imag_val*imag_val),
                               precision, atan2(imag_val, real_val));
                        break;
                }
                
                if (i < size - 1) printf(", ");
                if ((i + 1) % 8 == 0) printf("\n  ");
            }
            
            free(indices);
        }
    } else {
        // Real array - use existing print logic
        // For now, delegate to existing function
        // TODO: Update existing print to be type-aware
        printf("Type-aware printing for real arrays not yet implemented\n");
    }
}

