#!/usr/bin/env python3
import numpy as np

# Test array creation
n_samples = 5
x_vals = np.linspace(0.0, 1.0, n_samples)
y_vals = np.linspace(0.0, 1.0, n_samples)
X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')

# Different ways to create the array
points1 = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)])
points2 = np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]).astype(np.float64)
points3 = np.ascontiguousarray(np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]).astype(np.float64))

print("points1 dtype:", points1.dtype, "flags:", points1.flags)
print("points2 dtype:", points2.dtype, "flags:", points2.flags)
print("points3 dtype:", points3.dtype, "flags:", points3.flags)

# Make array read-only
points3.flags.writeable = False
print("points3 read-only flags:", points3.flags)
