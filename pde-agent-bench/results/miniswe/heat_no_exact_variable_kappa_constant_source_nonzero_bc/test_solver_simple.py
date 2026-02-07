#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import solver

# Mock the sampling function to return simple data
original_sample = solver.sample_solution_on_grid

def mock_sample(u_func, n_samples=50):
    import numpy as np
    x_vals = np.linspace(0.0, 1.0, n_samples)
    y_vals = np.linspace(0.0, 1.0, n_samples)
    u_grid = np.zeros((n_samples, n_samples))
    return x_vals, y_vals, u_grid

solver.sample_solution_on_grid = mock_sample

# Run with minimal parameters
import shutil
import os
if os.path.exists('test_output_simple'):
    shutil.rmtree('test_output_simple')
    
solver.main()
