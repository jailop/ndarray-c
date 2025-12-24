/**
 * File I/O operations for ndarrays
 */

#include "ndarray_internal.h"

// Magic number for ndarray binary files: "NDAR" in ASCII
#define NDARRAY_MAGIC 0x4E444152
#define NDARRAY_VERSION 1

int ndarray_save(NDArray arr, const char *filename) {
    assert(arr != NULL && "ndarray cannot be NULL");
    assert(filename != NULL && "filename cannot be NULL");
    
    FILE *fp = fopen(filename, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open file '%s' for writing\n", filename);
        return -1;
    }
    
    // Write magic number
    uint32_t magic = NDARRAY_MAGIC;
    if (fwrite(&magic, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "Error: Failed to write magic number\n");
        fclose(fp);
        return -1;
    }
    
    // Write version
    uint32_t version = NDARRAY_VERSION;
    if (fwrite(&version, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "Error: Failed to write version\n");
        fclose(fp);
        return -1;
    }
    
    // Write number of dimensions
    uint64_t ndim = (uint64_t)arr->ndim;
    if (fwrite(&ndim, sizeof(uint64_t), 1, fp) != 1) {
        fprintf(stderr, "Error: Failed to write ndim\n");
        fclose(fp);
        return -1;
    }
    
    // Write dimension sizes
    for (size_t i = 0; i < arr->ndim; ++i) {
        uint64_t dim = (uint64_t)arr->dims[i];
        if (fwrite(&dim, sizeof(uint64_t), 1, fp) != 1) {
            fprintf(stderr, "Error: Failed to write dimension %zu\n", i);
            fclose(fp);
            return -1;
        }
    }
    
    // Write data
    size_t size = ndarray_size(arr);
    if (fwrite(arr->data, sizeof(double), size, fp) != size) {
        fprintf(stderr, "Error: Failed to write data\n");
        fclose(fp);
        return -1;
    }
    
    fclose(fp);
    return 0;
}

NDArray ndarray_load(const char *filename) {
    assert(filename != NULL && "filename cannot be NULL");
    
    FILE *fp = fopen(filename, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error: Cannot open file '%s' for reading\n", filename);
        return NULL;
    }
    
    // Read and verify magic number
    uint32_t magic;
    if (fread(&magic, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "Error: Failed to read magic number\n");
        fclose(fp);
        return NULL;
    }
    if (magic != NDARRAY_MAGIC) {
        fprintf(stderr, "Error: Invalid file format (bad magic number: 0x%X)\n", magic);
        fclose(fp);
        return NULL;
    }
    
    // Read version
    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, fp) != 1) {
        fprintf(stderr, "Error: Failed to read version\n");
        fclose(fp);
        return NULL;
    }
    if (version != NDARRAY_VERSION) {
        fprintf(stderr, "Error: Unsupported file version: %u\n", version);
        fclose(fp);
        return NULL;
    }
    
    // Read number of dimensions
    uint64_t ndim;
    if (fread(&ndim, sizeof(uint64_t), 1, fp) != 1) {
        fprintf(stderr, "Error: Failed to read ndim\n");
        fclose(fp);
        return NULL;
    }
    if (ndim < 2) {
        fprintf(stderr, "Error: Invalid ndim (must be >= 2): %lu\n", ndim);
        fclose(fp);
        return NULL;
    }
    
    // Read dimension sizes
    size_t *dims = (size_t*)malloc(sizeof(size_t) * ((size_t)ndim + 1));
    if (dims == NULL) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        fclose(fp);
        return NULL;
    }
    
    for (size_t i = 0; i < (size_t)ndim; ++i) {
        uint64_t dim;
        if (fread(&dim, sizeof(uint64_t), 1, fp) != 1) {
            fprintf(stderr, "Error: Failed to read dimension %zu\n", i);
            free(dims);
            fclose(fp);
            return NULL;
        }
        dims[i] = (size_t)dim;
    }
    dims[(size_t)ndim] = 0;  // Sentinel
    
    // Create ndarray
    NDArray arr = ndarray_new(dims);
    free(dims);
    
    if (arr == NULL) {
        fprintf(stderr, "Error: Failed to create ndarray\n");
        fclose(fp);
        return NULL;
    }
    
    // Read data
    size_t size = ndarray_size(arr);
    if (fread(arr->data, sizeof(double), size, fp) != size) {
        fprintf(stderr, "Error: Failed to read data\n");
        ndarray_free(arr);
        fclose(fp);
        return NULL;
    }
    
    fclose(fp);
    return arr;
}
