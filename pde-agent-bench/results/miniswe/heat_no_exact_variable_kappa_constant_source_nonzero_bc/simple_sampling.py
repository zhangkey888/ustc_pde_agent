import numpy as np
from scipy.interpolate import griddata

def simple_sample_solution(u_func, n_samples=50):
    """Simple sampling by interpolating from mesh vertices to regular grid."""
    # Get mesh and solution values at vertices
    msh = u_func.function_space.mesh
    V = u_func.function_space
    
    # Get coordinates of all vertices
    coords = msh.geometry.x[:, :2]  # (n_vertices, 2)
    
    # Get solution values at vertices
    # For Lagrange P1, DOFs correspond to vertices
    u_vals = u_func.x.array
    
    # Create regular grid
    x_vals = np.linspace(0.0, 1.0, n_samples)
    y_vals = np.linspace(0.0, 1.0, n_samples)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Interpolate using nearest neighbor (simplest)
    # This will work even if scipy is not available
    u_grid = np.zeros((n_samples, n_samples))
    
    # Simple nearest neighbor interpolation
    for i in range(n_samples):
        for j in range(n_samples):
            x = X[i, j]
            y = Y[i, j]
            # Find closest vertex
            distances = np.sqrt((coords[:, 0] - x)**2 + (coords[:, 1] - y)**2)
            closest_idx = np.argmin(distances)
            u_grid[i, j] = u_vals[closest_idx]
    
    return x_vals, y_vals, u_grid
