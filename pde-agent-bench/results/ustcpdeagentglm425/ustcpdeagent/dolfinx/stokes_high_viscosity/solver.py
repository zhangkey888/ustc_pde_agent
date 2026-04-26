import os
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nu = float(case_spec["pde"]["parameters"]["nu"])
    
    out = case_spec["output"]
    nx_out = out["grid"]["nx"]
    ny_out = out["grid"]["ny"]
    bbox = out["grid"]["bbox"]
    
    mesh_res = 96
    
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    
    u_ex = ufl.as_vector([pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
                          -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])])
    p_ex = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    a = (2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx)
    
    f1 = (2.0 * nu * pi**3 - pi) * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    f2 = -(2.0 * nu * pi**3 + pi) * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    f_expr = ufl.as_vector([f1, f2])
    
    L = ufl.inner(f_expr, v) * ufl.dx
    
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    u_bc_func = fem.Function(V)
    u_bc_expr = ufl.as_vector([pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0]),
                               -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])])
    u_bc_func.interpolate(fem.Expression(u_bc_expr, V.element.interpolation_points))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    bcs_list = [bc_u]
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs_list.append(bc_p)
    
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs_list,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    
    w_h = problem.solve()
    w_h.x.scatter_forward()
    
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    u_h = w_h.sub(0).collapse()
    u_h.x.scatter_forward()
    
    M = ufl.inner(u_h - u_ex, u_h - u_ex) * ufl.dx
    M_form = fem.form(M)
    error_sq = MPI.COMM_WORLD.allreduce(fem.assemble_scalar(M_form), op=MPI.SUM)
    error_val = np.sqrt(max(error_sq, 0.0))
    
    if comm.rank == 0:
        print(f"[Stokes] mesh_res={mesh_res}, L2 vel error = {error_val:.6e}, iters={iterations}")
    
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        magnitude = np.linalg.norm(vals, axis=1)
        u_values[eval_map] = magnitude
    
    valid = np.isfinite(u_values).astype(np.float64)
    u_vals_clean = np.where(np.isfinite(u_values), u_values, 0.0)
    recv_vals = np.zeros_like(u_values)
    recv_valid = np.zeros_like(u_values)
    comm.Allreduce(u_vals_clean, recv_vals, op=MPI.SUM)
    comm.Allreduce(valid, recv_valid, op=MPI.SUM)
    u_values = np.where(recv_valid > 0, recv_vals / np.maximum(recv_valid, 1.0), 0.0)
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    pde = case_spec.get("pde", {})
    if "time" in pde:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
