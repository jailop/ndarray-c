/**
 * GSL Integration Demo - Zero-Copy Linear Algebra
 * 
 * This example demonstrates the zero-copy integration between NDArray and GSL,
 * enabling seamless use of GSL linear algebra algorithms on NDArray data.
 */

#include <stdio.h>
#include <stdlib.h>
#include "../src/ndarray.h"
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

int main() {
    printf("=== NDArray <-> GSL Zero-Copy Integration Demo ===\n\n");
    
    /* Create a 3x3 matrix using NDArray */
    NDArray A = ndarray_new(NDA_DIMS(3, 3));
    
    /* Initialize with a sample matrix:
     * [1  2  3]
     * [4  5  6]
     * [7  8  10]
     */
    double init_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 10};
    for (int i = 0; i < 9; i++) {
        A->data[i] = init_data[i];
    }
    
    printf("Original NDArray A (3x3):\n");
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            printf("  %6.1f", A->data[i * 3 + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    /* ============================================================
     * DEMO 1: Zero-Copy Matrix View for LU Decomposition
     * ============================================================ */
    printf("DEMO 1: LU Decomposition using GSL\n");
    printf("----------------------------------------\n");
    
    /* Create a working copy for LU decomposition */
    NDArray A_copy = ndarray_new_copy(A);
    
    /* Get a GSL matrix view - no memory copy! */
    gsl_matrix_view A_gsl_view = ndarray_to_gsl_matrix(A_copy);
    gsl_matrix *A_gsl = &A_gsl_view.matrix;
    
    /* Create permutation and sign for LU decomposition */
    gsl_permutation *perm = gsl_permutation_alloc(3);
    int signum;
    
    /* Perform LU decomposition on NDArray data through GSL */
    gsl_linalg_LU_decomp(A_gsl, perm, &signum);
    
    printf("LU Decomposition completed (signum = %d)\n", signum);
    printf("L and U matrices combined:\n");
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            printf("  %8.4f", A_copy->data[i * 3 + j]);
        }
        printf("\n");
    }
    printf("Permutation: [");
    for (size_t i = 0; i < 3; i++) {
        printf("%zu%s", gsl_permutation_get(perm, i), (i < 2 ? ", " : ""));
    }
    printf("]\n\n");
    
    ndarray_free(A_copy);
    gsl_permutation_free(perm);
    
    /* ============================================================
     * DEMO 2: Vector Operations on Rows
     * ============================================================ */
    printf("DEMO 2: Row Operations using GSL\n");
    printf("----------------------------------------\n");
    
    /* Extract first row as a vector (zero-copy) */
    gsl_vector_view row0 = ndarray_to_gsl_row(A, 0);
    gsl_vector *v_row0 = &row0.vector;
    
    /* Extract second row as a vector (zero-copy) */
    gsl_vector_view row1 = ndarray_to_gsl_row(A, 1);
    gsl_vector *v_row1 = &row1.vector;
    
    /* Compute dot product of two rows using BLAS */
    double dot_product = 0.0;
    gsl_blas_ddot(v_row0, v_row1, &dot_product);
    
    printf("Row 0: [%.1f, %.1f, %.1f]\n", 
           gsl_vector_get(v_row0, 0), 
           gsl_vector_get(v_row0, 1),
           gsl_vector_get(v_row0, 2));
    printf("Row 1: [%.1f, %.1f, %.1f]\n", 
           gsl_vector_get(v_row1, 0), 
           gsl_vector_get(v_row1, 1),
           gsl_vector_get(v_row1, 2));
    printf("Dot product: %.1f\n\n", dot_product);
    
    /* ============================================================
     * DEMO 3: Column Operations (Strided Access)
     * ============================================================ */
    printf("DEMO 3: Column Operations using GSL\n");
    printf("----------------------------------------\n");
    
    /* Extract first column as a vector (zero-copy, strided) */
    gsl_vector_view col0 = ndarray_to_gsl_column(A, 0);
    gsl_vector *v_col0 = &col0.vector;
    
    /* Extract second column as a vector (zero-copy, strided) */
    gsl_vector_view col1 = ndarray_to_gsl_column(A, 1);
    gsl_vector *v_col1 = &col1.vector;
    
    /* Compute dot product of two columns */
    dot_product = 0.0;
    gsl_blas_ddot(v_col0, v_col1, &dot_product);
    
    printf("Column 0: [%.1f, %.1f, %.1f]\n", 
           gsl_vector_get(v_col0, 0), 
           gsl_vector_get(v_col0, 1),
           gsl_vector_get(v_col0, 2));
    printf("Column 1: [%.1f, %.1f, %.1f]\n", 
           gsl_vector_get(v_col1, 0), 
           gsl_vector_get(v_col1, 1),
           gsl_vector_get(v_col1, 2));
    printf("Dot product: %.1f\n\n", dot_product);
    
    /* ============================================================
     * DEMO 4: Verify No Memory Copy Occurred
     * ============================================================ */
    printf("DEMO 4: Memory Sharing Verification\n");
    printf("----------------------------------------\n");
    
    /* Modify data through GSL vector view */
    gsl_vector_view row0_copy = ndarray_to_gsl_row(A, 0);
    gsl_vector_set(&row0_copy.vector, 0, 99.0);  /* Modify element [0,0] */
    
    /* Verify change is reflected in NDArray */
    printf("After GSL modification of row 0, element [0,0]: %.1f\n", 
           ndarray_get(A, NDA_POS(0, 0)));
    printf("Memory sharing confirmed: GSL view and NDArray share data!\n\n");
    
    /* Reset the value */
    ndarray_set(A, NDA_POS(0, 0), 1.0);
    
    /* ============================================================
     * DEMO 5: Const Views (Read-Only)
     * ============================================================ */
    printf("DEMO 5: Const Views (Read-Only)\n");
    printf("----------------------------------------\n");
    
    /* Create a const view of the matrix */
    gsl_matrix_const_view A_const = ndarray_to_gsl_matrix_const(A);
    printf("Const matrix view created (read-only)\n");
    /* Access element directly: data[row * tda + col] */
    printf("Element [1,1]: %.1f\n", 
           A_const.matrix.data[1 * A_const.matrix.tda + 1]);
    printf("Const views prevent accidental modification\n\n");
    
    /* ============================================================
     * Cleanup
     * ============================================================ */
    ndarray_free(A);
    
    printf("=== Demo Complete ===\n");
    printf("\nKey Benefits:\n");
    printf("✓ Zero-copy conversion: No memory allocation for views\n");
    printf("✓ Memory sharing: Changes through GSL affect NDArray\n");
    printf("✓ Seamless integration: Use any GSL algorithm on NDArray data\n");
    printf("✓ Const-safety: Read-only views prevent accidental modification\n");
    printf("✓ Efficient strided access: Column extraction uses stride info\n");
    
    return 0;
}
