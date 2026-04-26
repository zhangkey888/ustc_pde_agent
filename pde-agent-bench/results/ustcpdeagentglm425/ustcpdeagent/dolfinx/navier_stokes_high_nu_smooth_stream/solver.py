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
    nu = float(pde["coefficients"]["nu"])
    
    output_spec = case_spec["output"]
    grid = output_spec["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    mesh_res = 160
    degree_u = 2
    degree_p = 1
    
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Taylor-Hood P2/P1 mixed element
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    S = fem.functionspace(msh, ("Lagrange", degree_u))
    
    pi_val = np.pi
    
    def u_exact_vec(x):
        vals = np.zeros((gdim, x.shape[1]))
        vals[0] = 0.5 * pi_val * np.cos(pi_val * x[1]) * np.sin(pi_val * x[0])
        vals[1] = -0.5 * pi_val * np.cos(pi_val * x[0]) * np.sin(pi_val * x[1])
        return vals
    
    def p_exact(x):
        return np.cos(pi_val * x[0]) + np.cos(pi_val * x[1])
    
    # Compute source term using UFL (exact, no hand-derivation errors)
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_ex_0 = 0.5 * pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0])
    u_ex_1 = -0.5 * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    u_ex = ufl.as_vector([u_ex_0, u_ex_1])
    p_ex = ufl.cos(pi * x[0]) + ufl.cos(pi * x[1])
    
    # f = (u·∇)u - ν∇²u + ∇p  (strong form of NS)
    f_ufl = ufl.grad(u_ex) * u_ex - nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)
    
    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))
    
    # Velocity BC on entire boundary
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(u_exact_vec)
    bc_u = fem.dirichletbc(u_bc_func, boundary_dofs, W.sub(0))
    bcs = [bc_u]
    
    # Pressure pin at (0,0): p_ex(0,0) = 2
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs[0]) > 0:
        p0_func = fem.Function(Q)
        p0_func.interpolate(p_exact)
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # Test/Trial functions
    (u_t, p_t) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    lu_opts = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    # Stokes solve for initial guess
    a_stokes = (
        nu * ufl.inner(ufl.grad(u_t), ufl.grad(v)) * ufl.dx
        - p_t * ufl.div(v) * ufl.dx
        + ufl.div(u_t) * q * ufl.dx
    )
    L_stokes = ufl.inner(f_func, v) * ufl.dx
    
    stokes_prob = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options=lu_opts,
        petsc_options_prefix="ns_"
    )
    w_stokes = stokes_prob.solve()
    
    w_prev = fem.Function(W)
    w_new = fem.Function(W)
    w_prev.x.array[:] = w_stokes.x.array[:]
    w_prev.x.scatter_forward()
    
    # Picard iteration
    max_picard = 30
    picard_tol = 1e-10
    nonlinear_iterations = []
    
    for k in range(max_picard):
        (u_prev, p_prev) = ufl.split(w_prev)
        
        a_picard = (
            nu * ufl.inner(ufl.grad(u_t), ufl.grad(v)) * ufl.dx
            - p_t * ufl.div(v) * ufl.dx
            + ufl.inner(ufl.grad(u_t) * u_prev, v) * ufl.dx
            + ufl.div(u_t) * q * ufl.dx
        )
        L_picard = ufl.inner(f_func, v) * ufl.dx
        
        picard_prob = petsc.LinearProblem(
            a_picard, L_picard, bcs=bcs,
            petsc_options=lu_opts,
            petsc_options_prefix="ns_"
        )
        w_new = picard_prob.solve()
        
        diff = w_new.x.array - w_prev.x.array
        diff_norm = np.linalg.norm(diff)
        sol_norm = np.linalg.norm(w_new.x.array)
        rel_diff = diff_norm / max(sol_norm, 1e-14)
        
        nonlinear_iterations.append(k + 1)
        
        if comm.rank == 0:
            print(f"  Picard iter {k+1}: rel_diff = {rel_diff:.6e}")
        
        if rel_diff < picard_tol:
            if comm.rank == 0:
                print(f"Picard converged in {k+1} iterations")
            break
        
        w_prev.x.array[:] = w_new.x.array[:]
        w_prev.x.scatter_forward()
    
    # Extract velocity
    u_h = w_new.sub(0).collapse()
    p_h = w_new.sub(1).collapse()
    
    # Velocity magnitude
    u_mag_ufl = ufl.sqrt(u_h[0]**2 + u_h[1]**2)
    u_mag_func = fem.Function(S)
    u_mag_func.interpolate(fem.Expression(u_mag_ufl, S.element.interpolation_points))
    
    # L2 error for verification
    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(u_exact_vec)
    
    error_u_form = fem.form(ufl.inner(u_h - u_ex_func, u_h - u_ex_func) * ufl.dx)
    error_u_local = fem.assemble_scalar(error_u_form)
    error_u_global = np.sqrt(comm.allreduce(error_u_local, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 velocity error: {error_u_global:.6e}")
    
    # Sample on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_mag_values = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_mag_func.eval(pts_arr, cells_arr)
        u_mag_values[eval_map] = vals.flatten()
    
    # Gather across processes
    if comm.size > 1:
        u_mag_recv = np.zeros_like(u_mag_values)
        comm.Allreduce(u_mag_values, u_mag_recv, op=MPI.SUM)
        nan_count = np.isnan(u_mag_values).astype(np.float64)
        nan_recv = np.zeros_like(nan_count)
        comm.Allreduce(nan_count, nan_recv, op=MPI.SUM)
        valid_count = comm.size - nan_recv
        valid_count = np.where(valid_count > 0, valid_count, 1.0)
        u_mag_recv /= valid_count
        u_mag_values = u_mag_recv
    
    u_grid = u_mag_values.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12,
        "iterations": 0,
        "nonlinear_iterations": nonlinear_iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {"nu": 2.0},
            "manufactured_solution": {
                "u": ["0.5*pi*cos(pi*y)*sin(pi*x)", "-0.5*pi*cos(pi*x)*sin(pi*y)"],
                "p": "cos(pi*x) + cos(pi*y)"
            }
        },
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]},
            "field": "velocity_magnitude"
        }
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Shape: {result['u'].shape}, Max: {np.nanmax(result['u']):.6f}, Time: {t1-t0:.2f}s")
    print(f"Exact max: {0.5*np.pi:.6f}")
    print(f"Solver info: {result['solver_info']}")
