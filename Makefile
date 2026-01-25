LIB_NAME = ndarray
VERSION = 0.4.0
LIB_STATIC = lib$(LIB_NAME).a
LIB_SHARED = lib$(LIB_NAME).so.$(VERSION)
LIB_SHARED_MAJOR = lib$(LIB_NAME).so.1
LIB_SHARED_BASE = lib$(LIB_NAME).so


PREFIX ?= /usr/local
INCLUDEDIR = $(PREFIX)/include
LIBDIR = $(PREFIX)/lib


CFLAGS = -O3 -Wall -g -std=c99 -pedantic -march=native -fopenmp
CFLAGS_SHARED = $(CFLAGS) -fPIC
LDFLAGS = -lm -fopenmp -lopenblas

SRCDIR = src
TESTDIR = tests
BENCHDIR = benchmarks
EXAMPLEDIR = examples

SRCS = $(wildcard $(SRCDIR)/*.c)
TEST_SOURCES = $(filter-out $(TESTDIR)/test_ndarray.c, $(wildcard $(TESTDIR)/*.c))
EXAMPLE_SOURCES = $(wildcard $(EXAMPLEDIR)/*.c)

OBJ = $(SRCS:.c=.o)
OBJ_SHARED = $(SRCS:.c=_shared.o)
TEST_OBJECTS = $(TEST_SOURCES:.c=.o)
EXAMPLES = $(EXAMPLE_SOURCES:$(EXAMPLEDIR)/%.c=%)

.PHONY: all clean test benchmark install uninstall lib static shared docs examples

all: examples

lib: static shared

static: $(LIB_STATIC)

shared: $(LIB_SHARED)

test: ndarray_test
	./ndarray_test

benchmark:
	@$(BENCHDIR)/run_benchmark.sh

examples: $(EXAMPLES)

$(EXAMPLES): %: $(EXAMPLEDIR)/%.c $(OBJ)
	$(CC) $(CFLAGS) -o $@ $< $(OBJ) $(LDFLAGS)

docs:
	@echo "Generating C documentation with Doxygen..."
	@doxygen Doxyfile
	@echo "C documentation generated in docs/c-api/"
	@if command -v zig >/dev/null 2>&1; then \
		echo "Generating Zig documentation..."; \
		mkdir -p docs/zig-api; \
		zig build-lib -femit-docs=docs/zig-api -fno-emit-bin src/ndarray.zig 2>/dev/null || echo "Zig documentation generation skipped (may need dependencies)"; \
	else \
		echo "Zig not found, skipping Zig documentation"; \
	fi
	@echo "Documentation generated successfully"

install: lib
	install -d $(DESTDIR)$(LIBDIR)
	install -d $(DESTDIR)$(INCLUDEDIR)
	install -m 644 $(LIB_STATIC) $(DESTDIR)$(LIBDIR)/
	install -m 755 $(LIB_SHARED) $(DESTDIR)$(LIBDIR)/
	ln -sf $(LIB_SHARED) $(DESTDIR)$(LIBDIR)/$(LIB_SHARED_MAJOR)
	ln -sf $(LIB_SHARED_MAJOR) $(DESTDIR)$(LIBDIR)/$(LIB_SHARED_BASE)
	install -m 644 $(SRCDIR)/ndarray.h $(DESTDIR)$(INCLUDEDIR)/
	ldconfig -n $(DESTDIR)$(LIBDIR) 2>/dev/null || true

uninstall:
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_STATIC)
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_SHARED)
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_SHARED_MAJOR)
	rm -f $(DESTDIR)$(LIBDIR)/$(LIB_SHARED_BASE)
	rm -f $(DESTDIR)$(INCLUDEDIR)/ndarray.h

clean:
	rm -f $(EXAMPLES) ndarray_test $(OBJ) $(OBJ_SHARED) $(TEST_OBJECTS)
	rm -f $(LIB_STATIC) $(LIB_SHARED) $(LIB_SHARED_MAJOR) $(LIB_SHARED_BASE)
	rm -f benchmark_seq benchmark_omp benchmark_*.txt


$(LIB_STATIC): $(OBJ)
	ar rcs $@ $^

$(LIB_SHARED): $(OBJ_SHARED)
	$(CC) -shared -Wl,-soname,$(LIB_SHARED_MAJOR) -o $@ $^ $(LDFLAGS)


$(SRCDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(SRCDIR)/%_shared.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS_SHARED) -c $< -o $@

$(TESTDIR)/%.o: $(TESTDIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ -I$(SRCDIR)





ndarray_test: $(TEST_OBJECTS) $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -lcunit
