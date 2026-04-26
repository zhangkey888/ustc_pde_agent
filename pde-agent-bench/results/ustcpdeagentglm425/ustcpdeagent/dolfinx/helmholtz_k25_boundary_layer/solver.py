import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    """
    Solve Helmholtz equation: -∇²u - k²u = f in Ω, u = g on ∂Ω
    Manufactured solution: u = exp(4*x)*sin(π*y), k=25
    """
    # Extract parameters
    k_val = float(case_spec["pde"]["params"]["k"])  # 25.0
    
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox

    # ---- Numerical parameters ----
    mesh_resolution = 96
    element_degree = 3
    
    # Create mesh on unit square
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, mesh_resolution, mesh_resolution,
        cell_type=mesh.CellType.triangle
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Spatial coordinates for UFL expressions
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = exp(4x)*sin(πy)
    u_exact_ufl = ufl.exp(4 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # ∇²u = (16 - π²) * exp(4x)*sin(πy)
    # f = -∇²u - k²u = -(16 - π² + k²) * exp(4x)*sin(πy)
    f_coeff = -(16.0 - np.pi**2 + k_val**2)
    f_ufl = f_coeff * ufl.exp(4 * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Variational form: ∫(∇u·∇v - k²uv) dx = ∫fv dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx
    
    # Boundary conditions: u = g = exp(4x)*sin(πy) on entire ∂Ω
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve with direct LU (reliable for indefinite Helmholtz)
    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="helm_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get iteration info
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # ---- Sample solution onto output grid ----
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((pts.shape[1],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    # MPI gather
    comm = domain.comm
    local_for_max = np.where(np.isnan(u_values), -np.inf, u_values)
    global_for_max = np.zeros_like(local_for_max)
    comm.Allreduce(local_for_max, global_for_max, op=MPI.MAX)
    u_grid = global_for_max.reshape(ny_out, nx_out)
    
    # ---- Compute L2 error for verification ----
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    
    L2_error_ufl = ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
    L2_error_form = fem.form(L2_error_ufl)
    L2_error_local = fem.assemble_scalar(L2_error_form)
    L2_error = np.sqrt(domain.comm.allreduce(L2_error_local, op=MPI.SUM))
    
    if domain.comm.rank == 0:
        print(f"[Helmholtz] mesh_res={mesh_resolution}, P{element_degree}, k={k_val}")
        print(f"[Helmholtz] L2 error = {L2_error:.6e}")
        print(f"[Helmholtz] Linear iterations = {iterations}")
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": rtol,
        "iterations": int(iterations),
    }
    
    # Add time info if present
    pde_time = case_spec.get("pde", {}).get("time")
    if pde_time is not None:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
