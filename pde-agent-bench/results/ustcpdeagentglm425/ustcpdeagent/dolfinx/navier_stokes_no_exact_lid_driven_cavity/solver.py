import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    pde = case_spec["pde"]
    nu = float(pde["coefficients"]["viscosity"])
    
    grid_spec = case_spec["output"]["grid"]
    nx_grid = grid_spec["nx"]
    ny_grid = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    mesh_res = 256
    
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    fdim = msh.topology.dim - 1
    
    # Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # Unknown
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    # Residual
    F = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    
    J = ufl.derivative(F, w)
    
    # BCs
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
    
    u_zero = fem.Function(V)
    u_zero.interpolate(lambda x: np.vstack([np.zeros(x.shape[1]), np.zeros(x.shape[1])]))
    
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    right_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    right_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, right_facets)
    
    bc_top = fem.dirichletbc(u_top, top_dofs, W.sub(0))
    bc_bottom = fem.dirichletbc(u_zero, bottom_dofs, W.sub(0))
    bc_left = fem.dirichletbc(u_zero, left_dofs, W.sub(0))
    bc_right = fem.dirichletbc(u_zero, right_dofs, W.sub(0))
    
    bcs = [bc_top, bc_bottom, bc_left, bc_right]
    
    # Pressure pin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
    )
    if len(p_dofs) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))
    
    # Solve Stokes first
    (u_t, p_t) = ufl.TrialFunctions(W)
    a_s = (
        2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u_t)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p_t * ufl.div(v) * ufl.dx
        + ufl.div(u_t) * q * ufl.dx
    )
    L_s = ufl.inner(f, v) * ufl.dx
    
    stokes = petsc.LinearProblem(
        a_s, L_s, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        petsc_options_prefix="stokes_"
    )
    w_stokes = stokes.solve()
    
    # Initialize w with Stokes solution
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()
    
    # Now solve NS with NonlinearProblem
    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J,
        petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-8,
            "snes_atol": 1e-10,
            "snes_stol": 1e-12,
            "snes_max_it": 50,
            "snes_monitor": None,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_rtol": 1e-10,
            "ksp_max_it": 500,
        }
    )
    
    w_h = problem.solve()
    w.x.scatter_forward()
    
    # Get iteration info
    snes = problem._snes
    nl_its = snes.getIterationNumber()
    lin_its = snes.getLinearSolveIterations()
    
    # Extract velocity
    u_h = w_h.sub(0).collapse()
    
    # Check solution validity
    u_max = np.max(np.abs(u_h.x.array))
    if comm.rank == 0:
        print(f"Solution max abs value: {u_max:.6e}")
        print(f"Newton iterations: {nl_its}, Linear iterations: {lin_its}")
    
    # Sample on grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_grid)
    ys = np.linspace(ymin, ymax, ny_grid)
    XX, YY = np.meshgrid(xs, ys)
    
    pts = np.zeros((nx_grid * ny_grid, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.full((ny_grid, nx_grid), np.nan)
    if len(points_on_proc) > 0:
        u_vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        magnitudes = np.linalg.norm(u_vals, axis=1)
        flat = np.full((nx_grid * ny_grid,), np.nan)
        flat[eval_map] = magnitudes
        u_grid = flat.reshape(ny_grid, nx_grid)
    
    if comm.size > 1:
        u_safe = np.nan_to_num(u_grid, nan=0.0)
        u_result = np.zeros_like(u_safe)
        comm.Allreduce(u_safe, u_result, op=MPI.SUM)
        mask = np.logical_not(np.isnan(u_grid)).astype(float)
        mask_sum = np.zeros_like(mask)
        comm.Allreduce(mask, mask_sum, op=MPI.SUM)
        u_result[mask_sum == 0] = np.nan
        u_grid = u_result
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 2,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": lin_its,
            "nonlinear_iterations": [nl_its],
        }
    }
