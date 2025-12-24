# Library information
LIB_NAME = ndarray
VERSION = 1.0.0
LIB_STATIC = lib$(LIB_NAME).a
LIB_SHARED = lib$(LIB_NAME).so.$(VERSION)
LIB_SHARED_MAJOR = lib$(LIB_NAME).so.1
LIB_SHARED_BASE = lib$(LIB_NAME).so

# Installation directories
PREFIX ?= /usr/local
INCLUDEDIR = $(PREFIX)/include
LIBDIR = $(PREFIX)/lib

# Compiler flags
CFLAGS = -O3 -Wall -g -std=c99 -pedantic -march=native -fopenmp
CFLAGS_SHARED = $(CFLAGS) -fPIC
LDFLAGS = -lm -fopenmp -lopenblas
TEST_LDFLAGS = -lm -fopenmp -lopenblas -lcunit

# Targets
BIN = example
TEST_BIN = ndarray_test
BENCH_SEQ = benchmark_seq
BENCH_OMP = benchmark_omp

# Source files
SRCS = ndarray_core.c ndarray_creation.c ndarray_arithmetic.c \
       ndarray_linalg.c ndarray_manipulation.c ndarray_aggregation.c \
       ndarray_print.c ndarray_io.c

# Object files
OBJ = $(SRCS:.c=.o)
OBJ_SHARED = $(SRCS:.c=_shared.o)
TEST_OBJ = $(OBJ)
BENCH_OBJ = $(OBJ)

.PHONY: all clean test benchmark install uninstall lib static shared docs

all: $(BIN)

lib: static shared

static: $(LIB_STATIC)

shared: $(LIB_SHARED)

test: $(TEST_BIN)
	./$(TEST_BIN)

benchmark:
	@echo "Running benchmark comparison..."
	@./run_benchmark.sh

docs:
	@echo "Generating Doxygen documentation..."
	@cd .. && doxygen Doxyfile
	@echo "Documentation generated in ../docs/html/"
	@echo "Open ../docs/html/index.html in a browser to view"

install: lib
	install -d $(DESTDIR)$(LIBDIR)
	install -d $(DESTDIR)$(INCLUDEDIR)
	install -m 644 $(LIB_STATIC) $(DESTDIR)$(LIBDIR)/
	install -m 755 $(LIB_SHARED) $(DESTDIR)$(LIBDIR)/
	ln -sf $(LIB_SHARED) $(DESTDIR)$(LIBDIR)/$(LIB_SHARED_MAJOR)
	ln -sf $(LIB_SHARED_MAJOR) $(DESTDIR)$(LIBDIR)/$(LIB_SHARED_BASE)
	install -m 644 ndarray.h $(DESTDIR)$(INCLUDEDIR)/
	ldconfig -n $(DESTDIR)$(LIBDIR) 2>/dev/null || true

uninstall:
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_STATIC)
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_SHARED)
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_SHARED_MAJOR)
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_SHARED_BASE)
	rm -f $(DESTDIR)$(INCLUDEDIR)/ndarray.h

clean:
	rm -f $(BIN) $(TEST_BIN) $(OBJ) $(OBJ_SHARED) ndarray_test.o
	rm -f $(LIB_STATIC) $(LIB_SHARED) $(LIB_SHARED_MAJOR) $(LIB_SHARED_BASE)
	rm -f $(BENCH_SEQ) $(BENCH_OMP) benchmark_seq.txt benchmark_omp.txt
	rm -rf ../docs

# Library builds
$(LIB_STATIC): $(OBJ)
	ar rcs $@ $^

$(LIB_SHARED): $(OBJ_SHARED)
	$(CC) -shared -Wl,-soname,$(LIB_SHARED_MAJOR) -o $@ $^ $(LDFLAGS)

# Object files - static
%.o: %.c ndarray.h ndarray_internal.h
	$(CC) $(CFLAGS) -c $< -o $@

# Object files - shared
%_shared.o: %.c ndarray.h ndarray_internal.h
	$(CC) $(CFLAGS_SHARED) -c $< -o $@

$(BIN): $(OBJ) example.c
	$(CC) $(CFLAGS) -o $@ example.c $(OBJ) $(LDFLAGS)

$(TEST_BIN): ndarray_test.o $(TEST_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(TEST_LDFLAGS)

ndarray_test.o: ndarray_test.c ndarray.h
	$(CC) $(CFLAGS) -c ndarray_test.c

