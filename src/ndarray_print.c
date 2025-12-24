/**
 * Array printing functions
 */

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
    
    // Get terminal width
    struct winsize w;
    int term_width = 80; // Default fallback
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) != -1 && w.ws_col > 0) {
        term_width = w.ws_col;
    }
    
    // Print header
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
                printf("...\n");
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

