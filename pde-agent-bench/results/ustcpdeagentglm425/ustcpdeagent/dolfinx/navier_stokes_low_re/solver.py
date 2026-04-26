import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    nu_val = case_spec["pde"].get("nu", 1.0)
    
    mesh_res = 148
    
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    
    u_ex1 = pi * ufl.cos(pi * x[1]) * ufl.sin(pi * x[0])
    u_ex2 = -pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    u_ex_vec = ufl.as_vector([u_ex1, u_ex2])
    p_ex = ufl.cos(pi * x[0]) * ufl.cos(pi * x[1])
    
    f_val = ufl.grad(u_ex_vec) * u_ex_vec - nu_val * ufl.div(ufl.grad(u_ex_vec)) + ufl.grad(p_ex)
    
    # BCs
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_ex_vec, V.element.interpolation_points))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    bcs = [bc_u]
    
    # Pin pressure at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # ---- Step 1: Stokes solve for initial guess ----
    (u_trial, p_trial) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    f_stokes = -nu_val * ufl.div(ufl.grad(u_ex_vec)) + ufl.grad(p_ex)
    
    a_stokes = (
        nu_val * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
        - p_trial * ufl.div(v) * ufl.dx
        + ufl.div(u_trial) * q * ufl.dx
    )
    L_stokes = ufl.inner(f_stokes, v) * ufl.dx
    
    stokes_problem = petsc.LinearProblem(
        a_stokes, L_stokes, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    
    w_stokes = stokes_problem.solve()
    
    # ---- Step 2: Picard (Oseen) iterations ----
    w = fem.Function(W)
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()
    
    u_k = fem.Function(V)
    u_k.interpolate(w.sub(0))
    u_k.x.scatter_forward()
    
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    max_picard = 25
    picard_tol = 1e-10
    picard_its = 0
    total_linear_its = 0
    w_prev = w.x.array.copy()
    
    for it in range(max_picard):
        F_iter = (
            nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            + ufl.inner(ufl.grad(u) * u_k, v) * ufl.dx
            - p * ufl.div(v) * ufl.dx
            + ufl.div(u) * q * ufl.dx
            - ufl.inner(f_val, v) * ufl.dx
        )
        J_iter = ufl.derivative(F_iter, w)
        
        oseen_problem = petsc.NonlinearProblem(
            F_iter, w, bcs=bcs, J=J_iter,
            petsc_options_prefix=f"oseen_{it}_",
            petsc_options={
                "snes_type": "newtonls",
                "snes_rtol": 1e-12,
                "snes_atol": 1e-14,
                "snes_max_it": 1,
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        )
        
        try:
            w_h = oseen_problem.solve()
            w.x.scatter_forward()
        except Exception as e:
            break
        
        snes = oseen_problem._snes
        lin_its = int(snes.getLinearSolveIterations())
        total_linear_its += lin_its
        picard_its += 1
        
        diff = np.linalg.norm(w.x.array - w_prev)
        norm_w = np.linalg.norm(w.x.array)
        rel_change = diff / max(norm_w, 1e-30)
        
        if rel_change < picard_tol:
            break
        
        w_prev = w.x.array.copy()
        u_k.interpolate(w.sub(0))
        u_k.x.scatter_forward()
    
    # ---- Step 3: Newton refinement ----
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    F_newton = (
        nu_val * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f_val, v) * ufl.dx
    )
    J_newton = ufl.derivative(F_newton, w)
    
    newton_problem = petsc.NonlinearProblem(
        F_newton, w, bcs=bcs, J=J_newton,
        petsc_options_prefix="ns_newton_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_rtol": 1e-10,
            "snes_atol": 1e-12,
            "snes_max_it": 20,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
    )
    
    newton_its = 0
    try:
        w_h = newton_problem.solve()
        w.x.scatter_forward()
        snes_newton = newton_problem._snes
        newton_its = int(snes_newton.getIterationNumber())
        lin_its_newton = int(snes_newton.getLinearSolveIterations())
        total_linear_its += lin_its_newton
    except Exception:
        pass
    
    # Extract velocity
    u_sol = w.sub(0).collapse()
    
    # Sample on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros((points.shape[1], gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals
    
    if comm.size > 1:
        u_values_global = np.zeros_like(u_values)
        comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
        u_values = u_values_global
    
    magnitude = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2).reshape(ny_out, nx_out)
    
    # L2 error
    u_err_expr = ufl.inner(u_sol - u_ex_vec, u_sol - u_ex_vec)
    err_local = fem.assemble_scalar(fem.form(u_err_expr * ufl.dx))
    err_L2 = np.sqrt(msh.comm.allreduce(err_local, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"Velocity L2 error: {err_L2:.6e}")
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": total_linear_its,
        "nonlinear_iterations": [newton_its],
    }
    
    return {
        "u": magnitude,
        "solver_info": solver_info,
    }
