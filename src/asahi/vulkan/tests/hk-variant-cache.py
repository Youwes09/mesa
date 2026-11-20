#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run real cache round trips in one process and across processes."""
import os
import subprocess
import sys
import tempfile

binary = sys.argv[1]
env = {k: v for k, v in os.environ.items()
       if not k.startswith("MESA_SHADER_CACHE")}
with tempfile.TemporaryDirectory(prefix="hk-variant-cache-") as directory:
    env["MESA_SHADER_CACHE_DIR"] = directory
    subprocess.run([binary], env=env, check=True)
    subprocess.run([binary, "--cold-process"], env=env, check=True)
    subprocess.run([binary, "--warm-process"], env=env, check=True)

    print("cache round trips pass")
