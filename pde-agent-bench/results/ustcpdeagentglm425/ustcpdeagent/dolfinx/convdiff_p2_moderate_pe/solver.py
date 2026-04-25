import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    eps = pde["coefficients"]["epsilon"]
    beta = np.array(pde["coefficients"]["beta"])
    
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    mesh_res = 256
    elem_degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    def u_exact_expr(x):
        return np.sin(np.pi * x[0]) * np.sin(2 * np.pi * x[1])
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(u_exact_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    beta_vec = ufl.as_vector([PETSc.ScalarType(beta[0]), PETSc.ScalarType(beta[1])])
    
    pi_val = np.pi
    f_val = (eps * 5 * pi_val**2 * ufl.sin(pi_val * x[0]) * ufl.sin(2 * pi_val * x[1])
             + beta[0] * pi_val * ufl.cos(pi_val * x[0]) * ufl.sin(2 * pi_val * x[1])
             + beta[1] * 2 * pi_val * ufl.sin(pi_val * x[0]) * ufl.cos(2 * pi_val * x[1]))
    
    h = 1.0 / mesh_res
    beta_norm = np.sqrt(beta[0]**2 + beta[1]**2)
    Pe = beta_norm * h / (2.0 * eps)
    
    if Pe > 1.0:
        coth_Pe = 1.0 / np.tanh(Pe)
        tau_supg = h / (2.0 * beta_norm) * (coth_Pe - 1.0 / Pe)
        tau = fem.Constant(domain, PETSc.ScalarType(tau_supg))
        a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
             + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
             + tau * ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx)
        L = (ufl.inner(f_val, v) * ufl.dx
             + tau * ufl.inner(f_val, ufl.dot(beta_vec, ufl.grad(v))) * ufl.dx)
    else:
        a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
             + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx)
        L = ufl.inner(f_val, v) * ufl.dx
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": "1e-12",
            "ksp_atol": "1e-14",
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(u_exact_expr)
    error_L2 = domain.comm.allreduce(
        np.sqrt(fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx))),
        op=MPI.SUM
    )
    print(f"L2 error: {error_L2:.6e}")
    print(f"Element Pe: {Pe:.4f}, SUPG used: {Pe > 1.0}")
    
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    if comm.size > 1:
        u_values_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
        u_values = u_values_global
    
    nan_mask = np.isnan(u_values)
    if np.any(nan_mask):
        u_values[nan_mask] = 0.0
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-12,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}
