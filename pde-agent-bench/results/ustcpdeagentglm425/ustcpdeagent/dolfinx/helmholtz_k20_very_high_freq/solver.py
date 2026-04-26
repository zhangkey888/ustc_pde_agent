import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve the Helmholtz equation: -nabla^2(u) - k^2 * u = f in Omega, u = g on dOmega
    with manufactured solution u = sin(6*pi*x)*sin(5*pi*y), k = 20
    """
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec
    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    
    nx_grid = grid.get("nx", 64)
    ny_grid = grid.get("ny", 64)
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Get wavenumber
    k_val = pde.get("parameters", {}).get("k", 20.0)
    if isinstance(k_val, list):
        k_val = k_val[0]
    k = float(k_val)
    
    # Mesh resolution - fine mesh for k=20 to control pollution error
    # wavelength = 2*pi/20 ~ 0.314
    # h ~ 1/200 = 0.005, kh ~ 0.1, excellent resolution
    mesh_res = 200
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space - P4 for very high accuracy
    degree = 4
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define variational problem
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u_exact = sin(6*pi*x)*sin(5*pi*y)
    # Laplacian: -((6*pi)^2 + (5*pi)^2) * u_exact = -61*pi^2 * u_exact
    # f = -laplacian(u) - k^2*u = 61*pi^2 * u_exact - k^2 * u_exact = (61*pi^2 - k^2) * u_exact
    
    u_exact_expr = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])
    f_coeff = (61.0 * ufl.pi**2 - k**2) * u_exact_expr
    
    # Bilinear form: (grad u, grad v) - k^2*(u, v)
    a = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx \
        - k**2 * ufl.inner(u_trial, v_test) * ufl.dx
    
    # Linear form: (f, v)
    L = ufl.inner(f_coeff, v_test) * ufl.dx
    
    # Boundary conditions: u = g = sin(6*pi*x)*sin(5*pi*y) on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(
        fem.Expression(u_exact_expr, V.element.interpolation_points)
    )
    
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # Solve using direct LU (most robust for indefinite Helmholtz)
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
    
    # Get solver info
    ksp = problem.solver
    its = ksp.getIterationNumber()
    ksp_type_str = "preonly"
    pc_type_str = "lu"
    rtol_val = 1e-10
    
    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(
        fem.Expression(u_exact_expr, V.element.interpolation_points)
    )
    
    error_expr = ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
    error_form = fem.form(error_expr)
    l2_error_sq = fem.assemble_scalar(error_form)
    l2_error = float(np.sqrt(abs(float(l2_error_sq))))
    
    if comm.rank == 0:
        print(f"L2 error: {l2_error:.6e}", flush=True)
    
    # Sample solution onto output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_grid)
    ys = np.linspace(ymin, ymax, ny_grid)
    XX, YY = np.meshgrid(xs, ys)
    pts_flat = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_grid * ny_grid)])
    
    # Evaluate solution at grid points using geometry utilities
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_flat)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts_flat)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts_flat.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_flat[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros(nx_grid * ny_grid)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    # Gather from all processes
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
    
    u_grid = u_values_global.reshape(ny_grid, nx_grid)
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": ksp_type_str,
        "pc_type": pc_type_str,
        "rtol": rtol_val,
        "iterations": its,
        "l2_error": l2_error,
    }
    
    # Check if time info is needed
    if pde.get("time") is not None:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
    }
    
    return result
