version       = "0.4.2"
author        = "Jaime Lopez"
description   = "Nim bindings for ndarray-c library"
license       = "BSD-3-Clause"
srcDir        = "nim"
requires "nim >= 2.0"

task test, "Run all tests":
  exec "nim c -r tests/test_basic.nim"
  exec "nim c -r tests/test_operations.nim"
  exec "nim c -r tests/test_io.nim"

task docs, "Generate documentation":
  exec "mkdir -p docs"
  exec "nim doc --project --index:on --outdir:docs nim/ndarray.nim"
  echo "Documentation generated in docs/ directory"
  echo "Open docs/index.html in your browser to view"

task examples, "Build and run all examples":
  exec "for file in examples/*.nim; do echo \"Building $file...\"; nim c --path:src -r \"$file\" || exit 1; done"

task build_examples, "Build all examples without running":
  exec "for file in examples/*.nim; do echo \"Building $file...\"; nim c --path:src \"$file\" || exit 1; done"

task clean_examples, "Clean example executables":
  exec "rm -f examples/example_*"
  exec "rm -f examples/gbmpaths"

task clean, "Clean build artifacts":
  exec "rm -f tests/test_basic tests/test_operations tests/test_io"
  exec "rm -f test_ndarray_io.bin"
  exec "rm -f *.bin"
  exec "nimble clean_examples"
