#!/bin/sh
# Copyright 2026 The MuJoCo Metal contributors
# Licensed under the Apache License, Version 2.0.
#
# Bounded workspace-local rebuild entry point for the pinned instrumented
# MuJoCo trace build (plugin-descent + mesh-route MJCSDFTRACE records).
# Reconfigures (only if needed) and rebuilds the `mujoco` and `sdf_plugin`
# targets with the pinned compilers. Never touches Xcode selection,
# licensing, host settings, toolchains, or the installed MuJoCo reference.
# All inputs, build tree, and outputs stay inside this workspace.
#
# Usage: ./rebuild_trace.sh [build-dir]
#   Default build dir: .upstream/mujoco-trace/build-fixed
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(/usr/bin/dirname "$0")" && /bin/pwd -P)
WORKSPACE=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && /bin/pwd -P)
SRC="$WORKSPACE/.upstream/mujoco-trace"
BUILD="${1:-$SRC/build-fixed}"
LIB="$BUILD/lib/libmujoco.dylib"
PLUGIN="$BUILD/lib/libsdf_plugin.dylib"

need() {
  if ! /usr/bin/command -v "$1" >/dev/null 2>&1; then
    echo "rebuild_trace: missing required tool: $1" >&2
    exit 2
  fi
}
need cmake
[ -x /usr/bin/cc ] || { echo "rebuild_trace: missing /usr/bin/cc" >&2; exit 2; }
[ -x /usr/bin/c++ ] || { echo "rebuild_trace: missing /usr/bin/c++" >&2; exit 2; }
[ -f "$SRC/CMakeLists.txt" ] || {
  echo "rebuild_trace: missing pinned trace source: $SRC" >&2; exit 2; }

# The Xcode-beta toolchain path contains a space, which breaks CMake's
# internal make re-execution ("No such file or directory"). Use a
# space-free copy of the toolchain make for configuration and build (same
# binary, verified runnable; compilers stay pinned below).
SPACEFREE="$BUILD/spacefree-make"
/bin/mkdir -p "$BUILD"
case "$SPACEFREE" in *\ *) echo "rebuild_trace: build dir must not contain spaces" >&2; exit 2;; esac
if [ ! -x "$SPACEFREE" ] || ! "$SPACEFREE" --version >/dev/null 2>&1; then
  STAGED=""
  for candidate in \
      "/Applications/Xcode beta.app/Contents/Developer/usr/bin/make" \
      "$(/usr/bin/command -v make)"; do
    if [ -n "$candidate" ] && [ -x "$candidate" ]; then
      if /bin/cp "$candidate" "$SPACEFREE" 2>/dev/null \
          && /bin/chmod +x "$SPACEFREE" \
          && "$SPACEFREE" --version >/dev/null 2>&1; then
        STAGED="$candidate"
        echo "rebuild_trace: staged space-free make from $candidate"
        break
      fi
    fi
  done
  if [ -z "$STAGED" ]; then
    echo "rebuild_trace: no runnable space-free make available" >&2
    exit 2
  fi
fi

if [ ! -f "$BUILD/Makefile" ]; then
  cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DMUJOCO_BUILD_EXAMPLES=OFF \
    -DMUJOCO_BUILD_SIMULATE=OFF \
    -DMUJOCO_BUILD_TESTS=OFF \
    -DMUJOCO_BUILD_PYTHON=OFF \
    -DCMAKE_C_COMPILER=/usr/bin/cc \
    -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
    -DCMAKE_MAKE_PROGRAM="$SPACEFREE"
fi
MAKE="$SPACEFREE" /usr/bin/make -C "$BUILD" mujoco sdf_plugin -j4
for artifact in "$LIB" "$PLUGIN"; do
  [ -f "$artifact" ] || {
    echo "rebuild_trace: expected artifact missing: $artifact" >&2; exit 1; }
done
echo "rebuild_trace: OK"
echo "  lib:    $LIB"
echo "  plugin: $PLUGIN"
