switch("threads", "off")
switch("passL", "-lndarray")

when defined(ndarrayLibPath):
  switch("passL", "-L" & getEnv("NDARRAY_LIB_PATH", "/usr/local/lib"))
else:
  let ndarrayPath = getEnv("NDARRAY_LIB_PATH", "/usr/local/lib")
  switch("passL", "-L" & ndarrayPath)
