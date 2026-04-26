import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Helmholtz equation: -∇²u - k²u = f in Ω, u = 0 on ∂Ω
    """
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid_spec = output.get("grid", {})
    
    k_val = pde.get("k", 20.0)
    nx_out = grid_spec.get("nx", 64)
    ny_out = grid_spec.get("ny", 64)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Mesh resolution: N=448 P3 gives ~1.8M DOFs, good accuracy for k=20
    N = 448
    elem_degree = 3
    
    # Create mesh on unit square
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # Dirichlet BC: u = 0 on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    f_expr = 50.0 * ufl.exp(-200.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    
    # Weak form: (∇u·∇v) - k²(u·v) = (f·v)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Solve with LU direct solver (robust for indefinite systems)
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    iterations = 1
    rtol = 1e-10
    
    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    n_pts = nx_out * ny_out
    pts_flat = np.zeros((n_pts, 3))
    pts_flat[:, 0] = XX.ravel()
    pts_flat[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_flat)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts_flat)
    
    # Build cell and point arrays for evaluation
    cells = []
    pts_idx = []
    for i in range(n_pts):
        links = colliding_cells.links(i)
        if len(links) > 0:
            cells.append(links[0])
            pts_idx.append(i)
    
    u_values = np.zeros(n_pts)
    if len(cells) > 0:
        pts_eval = pts_flat[pts_idx]
        cells_arr = np.array(cells, dtype=np.int32)
        vals = u_sol.eval(pts_eval, cells_arr)
        for j, idx in enumerate(pts_idx):
            u_values[idx] = vals[j, 0]
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": elem_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}
