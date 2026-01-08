# Build script for ndarray-c on Windows with vcpkg
param(
    [string]$BuildType = "Release",
    [switch]$Clean,
    [switch]$Install,
    [string]$InstallPrefix = "install",
    [string]$VcpkgRoot = $env:VCPKG_ROOT,
    [string]$Generator = "Ninja"
)

$ErrorActionPreference = "Stop"

Write-Host "=== Building ndarray-c for Windows ===" -ForegroundColor Cyan
Write-Host "Build Type: $BuildType" -ForegroundColor Yellow
Write-Host "Generator: $Generator" -ForegroundColor Yellow

# Check for vcpkg
if (-not $VcpkgRoot) {
    # Try common locations
    $commonPaths = @("C:\vcpkg", "$HOME\vcpkg", "C:\src\vcpkg")
    foreach ($path in $commonPaths) {
        if (Test-Path "$path\vcpkg.exe") {
            $VcpkgRoot = $path
            break
        }
    }
}

if (-not $VcpkgRoot -or -not (Test-Path "$VcpkgRoot\vcpkg.exe")) {
    Write-Host "ERROR: vcpkg not found!" -ForegroundColor Red
    Write-Host "Please install vcpkg or set VCPKG_ROOT environment variable" -ForegroundColor Yellow
    Write-Host "See: https://github.com/microsoft/vcpkg" -ForegroundColor Cyan
    exit 1
}

Write-Host "Using vcpkg at: $VcpkgRoot" -ForegroundColor Green
$VcpkgToolchain = "$VcpkgRoot\scripts\buildsystems\vcpkg.cmake"

# Clean if requested
if ($Clean) {
    Write-Host "Cleaning build artifacts (preserving vcpkg dependencies)..." -ForegroundColor Yellow
    if (Test-Path "build") {
        # Remove everything except vcpkg_installed directory
        Get-ChildItem "build" -Exclude "vcpkg_installed" | ForEach-Object {
            Remove-Item -Recurse -Force $_.FullName
        }
    }
}

# Create build directory
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path "build" | Out-Null
}

# Step 1: Configure CMake (vcpkg will install dependencies automatically via manifest)
Write-Host "`n[1/3] Configuring CMake (vcpkg will install dependencies)..." -ForegroundColor Green

# Force GCC if available and not explicitly using MSVC generator
$cmakeArgs = @("-S", ".", "-B", "build", "-DCMAKE_TOOLCHAIN_FILE=$VcpkgToolchain", "-DCMAKE_BUILD_TYPE=$BuildType")

if ($Generator -eq "Ninja" -and (Get-Command gcc -ErrorAction SilentlyContinue)) {
    Write-Host "Using GCC with Ninja generator" -ForegroundColor Green
    $cmakeArgs += @("-G", "Ninja", "-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++")
} elseif ($Generator -eq "MinGW Makefiles") {
    Write-Host "Using GCC with MinGW Makefiles generator" -ForegroundColor Green
    $cmakeArgs += @("-G", "MinGW Makefiles", "-DCMAKE_C_COMPILER=gcc", "-DCMAKE_CXX_COMPILER=g++")
} else {
    $cmakeArgs += @("-G", $Generator)
}

& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    exit 1
}

# Step 2: Build
Write-Host "`n[2/3] Building..." -ForegroundColor Green
cmake --build build --config $BuildType
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Step 3: Install (optional)
if ($Install) {
    Write-Host "`n[3/3] Installing to $InstallPrefix..." -ForegroundColor Green
    cmake --install build --prefix $InstallPrefix --config $BuildType
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installation failed!" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n=== Build completed successfully! ===" -ForegroundColor Cyan
Write-Host "Binaries are in: build\$BuildType\" -ForegroundColor Yellow

# Show what was built
if (Test-Path "build\$BuildType") {
    Write-Host "`nBuilt files:" -ForegroundColor Yellow
    Get-ChildItem "build\$BuildType\*.exe", "build\$BuildType\*.dll", "build\$BuildType\*.lib" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "  - $($_.Name)" -ForegroundColor Gray
    }
}

Write-Host "`nTo run the example:" -ForegroundColor Cyan
Write-Host "  .\build\$BuildType\example.exe" -ForegroundColor White

