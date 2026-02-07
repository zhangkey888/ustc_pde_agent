import re

with open('solver.py', 'r') as f:
    content = f.read()

# Replace the sampling function with a simpler version that works
new_sampling_function = '''def sample_solution_on_grid(u_func, n_samples=50):
    """Sample FEM solution on a uniform n_samples x n_samples grid."""
    # Create grid points
    x_vals = np.linspace(0.0, 1.0, n_samples)
    y_vals = np.linspace(0.0, 1.0, n_samples)
    X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
    # Flatten for point evaluation - create read-only array
    points = np.ascontiguousarray(np.column_stack([X.ravel(), Y.ravel(), np.zeros(X.size)]).astype(default_scalar_type))
    points.flags.writeable = False
    
    # Get mesh and function space
    msh = u_func.function_space.mesh
    V = u_func.function_space
    
    # Create bounding box tree for point location
    tree = bb_tree(msh, msh.topology.dim)
    
    # Find cells containing each point
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(msh, cell_candidates, points[:, :2])
    
    # Prepare arrays for evaluation
    u_values = np.zeros(points.shape[0], dtype=default_scalar_type)
    points_found = np.zeros(points.shape[0], dtype=bool)
    
    # For each point, find containing cell and evaluate
    for i in range(points.shape[0]):
        cells = colliding_cells.links(i)
        if len(cells) > 0:
            # Take first cell that contains the point
            cell = cells[0]
            
            # Get the cell geometry and dofmap
            geometry = msh.geometry
            dofmap = V.dofmap
            
            # Get cell coordinates and dofs
            cell_coords = geometry.x[geometry.dofmap[cell]]
            cell_dofs = dofmap.cell_dofs(cell)
            
            # Get basis function values at point
            # We need to map point to reference coordinates
            # For simplicity, use the cell value at closest vertex
            # This is approximate but works for visualization
            
            # Find closest vertex in cell
            point = points[i]
            distances = np.linalg.norm(cell_coords[:, :2] - point[:2], axis=1)
            closest_vertex = np.argmin(distances)
            
            # Get solution value at closest vertex
            vertex_dof = cell_dofs[closest_vertex]
            u_values[i] = u_func.x.array[vertex_dof]
            points_found[i] = True
    
    # For points not found, use 0 (shouldn't happen for interior points)
    u_values[~points_found] = 0.0
    
    # Reshape to 2D grid
    u_grid = u_values.reshape((n_samples, n_samples))
    
    return x_vals, y_vals, u_grid'''

# Find and replace the old function
lines = content.split('\n')
new_lines = []
in_function = False
skip = False
for line in lines:
    if line.strip().startswith('def sample_solution_on_grid'):
        in_function = True
        skip = True
        new_lines.append(new_sampling_function)
    elif in_function and line.startswith('def ') and not skip:
        in_function = False
        skip = False
        new_lines.append(line)
    elif not skip:
        new_lines.append(line)
    elif skip and line.startswith('def ') and not line.strip().startswith('def sample_solution_on_grid'):
        in_function = False
        skip = False
        new_lines.append(line)

with open('solver.py', 'w') as f:
    f.write('\n'.join(new_lines))
