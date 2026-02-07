# Nim script configuration
switch("threads", "off")
switch("passL", "-lndarray")

# Allow configurable ndarray-c library path
# Priority: command-line --passL flags > environment variable > default
when defined(ndarrayLibPath):
  switch("passL", "-L" & getEnv("NDARRAY_LIB_PATH", "/usr/local/lib"))
else:
  let ndarrayPath = getEnv("NDARRAY_LIB_PATH", "/usr/local/lib")
  switch("passL", "-L" & ndarrayPath)

