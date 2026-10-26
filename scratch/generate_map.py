#!/usr/bin/env python3
import subprocess
from pathlib import Path
from collections import defaultdict

files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "HEAD"], text=True).splitlines()

# Build tree dictionary
tree = lambda: defaultdict(tree)
root = tree()

for path in files:
    parts = path.split("/")
    curr = root
    for p in parts:
        curr = curr[p]

print(f"Total files in HEAD: {len(files)}")
