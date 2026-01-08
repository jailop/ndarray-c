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

# Enable Fortran if available (needed for OpenBLAS with Fortran OpenMP on some systems)
include(CheckLanguage)
check_language(Fortran)
if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_STATIC_LIBS "Build static libraries" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build C test suite (requires CUnit)" ON)
option(BUILD_BENCHMARKS "Build benchmark programs" ON)

# Compiler-specific flags
if(MSVC)
    # MSVC specific flags (Note: MSVC doesn't support C99 VLAs, use GCC/MinGW instead)
    add_compile_options(/W3 /O2)
    message(WARNING "MSVC detected - this project uses C99 VLAs which MSVC doesn't support. Please use GCC (MinGW-w64) or Clang instead.")
else()
    # GCC/Clang/MinGW flags
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -pedantic")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    endif()
endif()

# OpenMP Configuration
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

# OpenMP is disabled on MSVC due to OpenMP 2.0 limitations
# On Windows with GCC/MinGW, OpenMP works fine; on Linux/macOS it's required
if(MSVC)
    message(STATUS "OpenMP disabled on MSVC (OpenMP 2.0 has limitations with C99 code)")
    set(OpenMP_C_FOUND FALSE)
elseif(WIN32)
    # On Windows with MinGW/GCC, try to find OpenMP
    find_package(OpenMP)
else()
    find_package(OpenMP REQUIRED)
endif()

if(OpenMP_C_FOUND)
    set(OPENMP_LIBRARIES ${OpenMP_C_LIBRARIES})
    message(STATUS "OpenMP found and will be used")
else()
    if(NOT MSVC AND NOT WIN32)
        message(FATAL_ERROR "OpenMP is required but not found. On macOS: brew install libomp")
    else()
        message(WARNING "OpenMP not found or disabled. Building without OpenMP support.")
    endif()
endif()

# BLAS/OpenBLAS Configuration
# First try to use vcpkg or system-provided packages
find_package(OpenBLAS QUIET)
if(OpenBLAS_FOUND)
    message(STATUS "Found OpenBLAS via vcpkg or system")
    set(BLAS_LIBRARIES OpenBLAS::OpenBLAS)
else()
    # Try standard BLAS package
    find_package(BLAS QUIET)
    if(BLAS_FOUND)
        message(STATUS "Found BLAS libraries: ${BLAS_LIBRARIES}")
    else()
        # Platform-specific fallback for Homebrew on macOS
        if(APPLE)
            execute_process(
                COMMAND brew --prefix openblas
                OUTPUT_VARIABLE HOMEBREW_OPENBLAS_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(HOMEBREW_OPENBLAS_PREFIX)
                set(BLAS_LIBRARIES "${HOMEBREW_OPENBLAS_PREFIX}/lib/libopenblas.dylib")
                set(OPENBLAS_INCLUDE_DIR "${HOMEBREW_OPENBLAS_PREFIX}/include")
                message(STATUS "Using Homebrew OpenBLAS: ${HOMEBREW_OPENBLAS_PREFIX}")
            endif()
        endif()
        
        if(NOT BLAS_LIBRARIES)
            message(FATAL_ERROR "BLAS library not found. On Windows, install via vcpkg. On macOS: brew install openblas. On Linux: install libopenblas-dev.")
        endif()
    endif()
endif()

# OpenBLAS include directory detection
if(NOT OPENBLAS_INCLUDE_DIR AND NOT OpenBLAS_FOUND)
    if(WIN32)
        # On Windows with vcpkg, headers should be in the include path automatically
        find_path(OPENBLAS_INCLUDE_DIR cblas.h)
    elseif(APPLE)
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
    
    if(OPENBLAS_INCLUDE_DIR)
        message(STATUS "Found OpenBLAS include directory: ${OPENBLAS_INCLUDE_DIR}")
    else()
        message(WARNING "Could not find OpenBLAS include directory - may cause build issues")
    endif()
endif()

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
    src/ndarray_random.c
)

set(NDARRAY_HEADERS
    src/ndarray.h
    src/ndarray_internal.h
)

add_library(ndarray_obj OBJECT ${NDARRAY_SOURCES})
target_include_directories(ndarray_obj PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)
if(OPENBLAS_INCLUDE_DIR)
    target_include_directories(ndarray_obj PRIVATE ${OPENBLAS_INCLUDE_DIR})
endif()
# Get include directories from OpenBLAS target if using vcpkg/CMake config
if(OpenBLAS_FOUND AND TARGET OpenBLAS::OpenBLAS)
    get_target_property(OPENBLAS_INCLUDE_DIRS OpenBLAS::OpenBLAS INTERFACE_INCLUDE_DIRECTORIES)
    if(OPENBLAS_INCLUDE_DIRS)
        target_include_directories(ndarray_obj PRIVATE ${OPENBLAS_INCLUDE_DIRS})
    endif()
endif()
if(NOT MSVC)
    target_compile_options(ndarray_obj PRIVATE ${OpenMP_C_FLAGS})
endif()
set_property(TARGET ndarray_obj PROPERTY POSITION_INDEPENDENT_CODE ON)

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
    
    # Link libraries
    target_link_libraries(ndarray_static PUBLIC ${BLAS_LIBRARIES})
    
    # Link math library on Unix-like systems
    if(UNIX AND NOT APPLE)
        target_link_libraries(ndarray_static PUBLIC m)
    endif()
    
    # Link OpenMP if available
    if(OpenMP_C_FOUND)
        target_link_libraries(ndarray_static PUBLIC OpenMP::OpenMP_C)
    endif()
endif()

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
    
    # Link libraries
    target_link_libraries(ndarray_shared PUBLIC ${BLAS_LIBRARIES})
    
    # Link math library on Unix-like systems
    if(UNIX AND NOT APPLE)
        target_link_libraries(ndarray_shared PUBLIC m)
    endif()
    
    # Link OpenMP if available
    if(OpenMP_C_FOUND)
        target_link_libraries(ndarray_shared PUBLIC OpenMP::OpenMP_C)
    endif()
endif()

if(BUILD_EXAMPLES)
    add_executable(example examples/example.c)
    if(TARGET ndarray_static)
        target_link_libraries(example PRIVATE ndarray_static)
    else()
        target_link_libraries(example PRIVATE ndarray_shared)
    endif()
endif()

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

if(BUILD_BENCHMARKS)
    # Sequential benchmark (without OpenMP)
    add_executable(benchmark_seq benchmarks/benchmark.c)
    if(MSVC)
        target_compile_options(benchmark_seq PRIVATE /O2 /W3)
    else()
        target_compile_options(benchmark_seq PRIVATE -O3 -Wall -pedantic)
    endif()
    target_include_directories(benchmark_seq PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(OPENBLAS_INCLUDE_DIR)
        target_include_directories(benchmark_seq PRIVATE ${OPENBLAS_INCLUDE_DIR})
    endif()
    if(TARGET ndarray_static)
        # Link with a modified static lib without OpenMP
        target_sources(benchmark_seq PRIVATE ${NDARRAY_SOURCES})
        target_link_libraries(benchmark_seq PRIVATE ${BLAS_LIBRARIES})
        if(UNIX AND NOT APPLE)
            target_link_libraries(benchmark_seq PRIVATE m)
        endif()
    else()
        target_link_libraries(benchmark_seq PRIVATE ndarray_shared)
        if(UNIX AND NOT APPLE)
            target_link_libraries(benchmark_seq PRIVATE m)
        endif()
    endif()
    
    # OpenMP benchmark (with OpenMP)
    add_executable(benchmark_omp benchmarks/benchmark.c)
    if(MSVC)
        target_compile_options(benchmark_omp PRIVATE /openmp /O2 /W3)
    else()
        target_compile_options(benchmark_omp PRIVATE ${OpenMP_C_FLAGS} -O3 -Wall -pedantic)
    endif()
    target_include_directories(benchmark_omp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    if(OPENBLAS_INCLUDE_DIR)
        target_include_directories(benchmark_omp PRIVATE ${OPENBLAS_INCLUDE_DIR})
    endif()
    if(TARGET ndarray_static)
        target_link_libraries(benchmark_omp PRIVATE ndarray_static)
        if(OpenMP_C_FOUND)
            target_link_libraries(benchmark_omp PRIVATE OpenMP::OpenMP_C)
        endif()
    else()
        target_link_libraries(benchmark_omp PRIVATE ndarray_shared)
        if(OpenMP_C_FOUND)
            target_link_libraries(benchmark_omp PRIVATE OpenMP::OpenMP_C)
        endif()
    endif()
    install(TARGETS benchmark_seq benchmark_omp
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

include(GNUInstallDirs)

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

install(FILES src/ndarray.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(BUILD_EXAMPLES)
    install(TARGETS example
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

install(EXPORT ndarrayTargets
    FILE ndarrayTargets.cmake
    NAMESPACE ndarray::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ndarray
)

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
