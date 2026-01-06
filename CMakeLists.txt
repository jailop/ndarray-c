cmake_minimum_required(VERSION 3.15)

if(APPLE AND NOT DEFINED CMAKE_C_COMPILER)
    find_program(GCC_COMPILER 
        NAMES gcc-15 gcc-14 gcc-13 gcc-12 gcc-11 gcc
        PATHS /opt/homebrew/bin /usr/local/bin
        NO_DEFAULT_PATH
    )
    if(GCC_COMPILER)
        set(CMAKE_C_COMPILER ${GCC_COMPILER})
        message(STATUS "Found GCC: ${GCC_COMPILER}")
    else()
        message(WARNING "GCC not found, will use system default compiler")
    endif()
endif()

project(ndarray VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build C test suite (requires CUnit)" ON)
option(BUILD_BENCHMARKS "Build benchmark programs" ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -pedantic")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
endif()

if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(HOMEBREW_LIBOMP_PREFIX)
        # Check if using GCC or Clang
        if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
            # GCC supports OpenMP natively
            set(OpenMP_C_FLAGS "-fopenmp")
            set(OpenMP_C_LIB_NAMES "gomp")
            set(OpenMP_gomp_LIBRARY "gomp")
        else()
            # Clang needs special flags
            set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
            set(OpenMP_C_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib")
        endif()
    endif()
endif()

find_package(OpenMP REQUIRED)
if(OpenMP_C_FOUND)
    set(OPENMP_LIBRARIES ${OpenMP_C_LIBRARIES})
else()
    message(FATAL_ERROR "OpenMP is required but not found. On macOS: brew install libomp")
endif()

# Find OpenBLAS - prefer Homebrew installation
if(APPLE)
    # First try Homebrew OpenBLAS
    execute_process(
        COMMAND brew --prefix openblas
        OUTPUT_VARIABLE HOMEBREW_OPENBLAS_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(HOMEBREW_OPENBLAS_PREFIX)
        set(BLA_VENDOR OpenBLAS)
        set(BLAS_LIBRARIES "${HOMEBREW_OPENBLAS_PREFIX}/lib/libopenblas.dylib")
        set(OPENBLAS_INCLUDE_DIR "${HOMEBREW_OPENBLAS_PREFIX}/include")
        message(STATUS "Using Homebrew OpenBLAS: ${HOMEBREW_OPENBLAS_PREFIX}")
    endif()
endif()

if(NOT BLAS_LIBRARIES)
    find_package(BLAS REQUIRED)
endif()

# Platform-specific OpenBLAS header detection
if(NOT OPENBLAS_INCLUDE_DIR)
    if(APPLE)
        # On macOS (Homebrew), cblas.h is directly in include
        find_path(OPENBLAS_INCLUDE_DIR cblas.h
            HINTS
            /opt/homebrew/opt/openblas/include
            /usr/local/opt/openblas/include
        )
    else()
        # On Linux, it might be in openblas/ subdirectory
        find_path(OPENBLAS_INCLUDE_DIR openblas/cblas.h
            HINTS
            /usr/include
            /usr/local/include
        )
        if(NOT OPENBLAS_INCLUDE_DIR)
            find_path(OPENBLAS_INCLUDE_DIR cblas.h
                HINTS
                /usr/include/openblas
                /usr/local/include/openblas
            )
        endif()
    endif()
endif()

if(OPENBLAS_INCLUDE_DIR)
    message(STATUS "Found OpenBLAS include directory: ${OPENBLAS_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Could not find OpenBLAS include directory")
endif()

# Library source files
set(NDARRAY_SOURCES
    src/ndarray_core.c
    src/ndarray_creation.c
    src/ndarray_arithmetic.c
    src/ndarray_linalg.c
    src/ndarray_manipulation.c
    src/ndarray_aggregation.c
    src/ndarray_print.c
    src/ndarray_io.c
    src/ndarray_comparison.c
)

set(NDARRAY_HEADERS
    src/ndarray.h
    src/ndarray_internal.h
)

# Create object library for common compilation
add_library(ndarray_obj OBJECT ${NDARRAY_SOURCES})
target_include_directories(ndarray_obj PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
target_include_directories(ndarray_obj PRIVATE ${OPENBLAS_INCLUDE_DIR})
target_compile_options(ndarray_obj PRIVATE ${OpenMP_C_FLAGS})
set_property(TARGET ndarray_obj PROPERTY POSITION_INDEPENDENT_CODE ON)

# Static library
if(BUILD_STATIC_LIBS)
    add_library(ndarray_static STATIC $<TARGET_OBJECTS:ndarray_obj>)
    set_target_properties(ndarray_static PROPERTIES
        OUTPUT_NAME ndarray
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
    )
    target_include_directories(ndarray_static PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(ndarray_static PUBLIC
        m
        ${BLAS_LIBRARIES}
    )
    if(OpenMP_C_FOUND)
        target_link_libraries(ndarray_static PUBLIC OpenMP::OpenMP_C)
    endif()
endif()

# Shared library
if(BUILD_SHARED_LIBS)
    add_library(ndarray_shared SHARED $<TARGET_OBJECTS:ndarray_obj>)
    set_target_properties(ndarray_shared PROPERTIES
        OUTPUT_NAME ndarray
        VERSION ${PROJECT_VERSION}
        SOVERSION 1
    )
    target_include_directories(ndarray_shared PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(ndarray_shared PUBLIC
        m
        ${BLAS_LIBRARIES}
    )
    if(OpenMP_C_FOUND)
        target_link_libraries(ndarray_shared PUBLIC OpenMP::OpenMP_C)
    endif()
endif()

# Example executable
if(BUILD_EXAMPLES)
    add_executable(example examples/example.c)
    if(TARGET ndarray_static)
        target_link_libraries(example PRIVATE ndarray_static)
    else()
        target_link_libraries(example PRIVATE ndarray_shared)
    endif()
endif()

# Test executable
if(BUILD_TESTS)
    # Find CUnit with Homebrew hint on macOS
    if(APPLE AND HOMEBREW_OPENBLAS_PREFIX)
        execute_process(
            COMMAND brew --prefix cunit
            OUTPUT_VARIABLE HOMEBREW_CUNIT_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(HOMEBREW_CUNIT_PREFIX)
            set(CMAKE_PREFIX_PATH "${HOMEBREW_CUNIT_PREFIX}" ${CMAKE_PREFIX_PATH})
        endif()
    endif()
    
    find_library(CUNIT_LIBRARY NAMES cunit CUnit
        HINTS ${HOMEBREW_CUNIT_PREFIX}/lib
    )
    find_path(CUNIT_INCLUDE_DIR CUnit/CUnit.h
        HINTS ${HOMEBREW_CUNIT_PREFIX}/include
    )
    
    if(CUNIT_LIBRARY AND CUNIT_INCLUDE_DIR)
        message(STATUS "Found CUnit: ${CUNIT_LIBRARY}")
        add_executable(ndarray_test tests/test_ndarray.c)
        target_include_directories(ndarray_test PRIVATE ${CUNIT_INCLUDE_DIR})
        if(TARGET ndarray_static)
            target_link_libraries(ndarray_test PRIVATE ndarray_static ${CUNIT_LIBRARY})
        else()
            target_link_libraries(ndarray_test PRIVATE ndarray_shared ${CUNIT_LIBRARY})
        endif()
        
        enable_testing()
        add_test(NAME ndarray_test COMMAND ndarray_test)
    else()
        message(WARNING "CUnit library not found - skipping test build. Install with: brew install cunit")
    endif()
endif()

# Benchmark executables
if(BUILD_BENCHMARKS)
    # Sequential benchmark (without OpenMP)
    add_executable(benchmark_seq benchmarks/benchmark.c)
    target_compile_options(benchmark_seq PRIVATE -O3 -Wall -pedantic)
    target_include_directories(benchmark_seq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${OPENBLAS_INCLUDE_DIR})
    if(TARGET ndarray_static)
        # Link with a modified static lib without OpenMP
        target_sources(benchmark_seq PRIVATE ${NDARRAY_SOURCES})
        target_link_libraries(benchmark_seq PRIVATE m ${BLAS_LIBRARIES})
    else()
        target_link_libraries(benchmark_seq PRIVATE ndarray_shared m)
    endif()
    
    # OpenMP benchmark (with OpenMP)
    add_executable(benchmark_omp benchmarks/benchmark.c)
    target_compile_options(benchmark_omp PRIVATE ${OpenMP_C_FLAGS} -O3 -Wall -pedantic)
    target_include_directories(benchmark_omp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${OPENBLAS_INCLUDE_DIR})
    if(TARGET ndarray_static)
        target_link_libraries(benchmark_omp PRIVATE ndarray_static OpenMP::OpenMP_C)
    else()
        target_link_libraries(benchmark_omp PRIVATE ndarray_shared OpenMP::OpenMP_C)
    endif()
    
    # Install benchmarks
    install(TARGETS benchmark_seq benchmark_omp
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# Installation
include(GNUInstallDirs)

# Install libraries
if(BUILD_STATIC_LIBS)
    install(TARGETS ndarray_static
        EXPORT ndarrayTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

if(BUILD_SHARED_LIBS)
    install(TARGETS ndarray_shared
        EXPORT ndarrayTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# Install headers
install(FILES src/ndarray.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install examples
if(BUILD_EXAMPLES)
    install(TARGETS example
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# Export targets for find_package support
install(EXPORT ndarrayTargets
    FILE ndarrayTargets.cmake
    NAMESPACE ndarray::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ndarray
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ndarrayConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ndarrayConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ndarrayConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ndarray
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ndarrayConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ndarrayConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ndarray
)

# Print configuration summary
message(STATUS "")
message(STATUS "========== Build Configuration ==========")
message(STATUS "Build type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "C Compiler:         ${CMAKE_C_COMPILER}")
message(STATUS "C Flags:            ${CMAKE_C_FLAGS}")
message(STATUS "Build shared libs:  ${BUILD_SHARED_LIBS}")
message(STATUS "Build static libs:  ${BUILD_STATIC_LIBS}")
message(STATUS "Build examples:     ${BUILD_EXAMPLES}")
message(STATUS "Build tests:        ${BUILD_TESTS}")
message(STATUS "Build benchmarks:   ${BUILD_BENCHMARKS}")
if(OpenMP_C_FOUND)
    message(STATUS "OpenMP version:     ${OpenMP_C_VERSION}")
endif()
message(STATUS "BLAS libraries:     ${BLAS_LIBRARIES}")
message(STATUS "Install prefix:     ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=========================================")
message(STATUS "")
