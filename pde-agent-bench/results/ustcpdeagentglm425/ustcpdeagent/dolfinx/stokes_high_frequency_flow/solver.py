import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

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
    nu = 1.0
    N = 256
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    mixed_el = basix_mixed_element([vel_el, pres_el])
    W = fem.functionspace(msh, mixed_el)
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_ex = ufl.as_vector([
        2*pi*ufl.cos(2*pi*x[1])*ufl.sin(2*pi*x[0]),
        -2*pi*ufl.cos(2*pi*x[0])*ufl.sin(2*pi*x[1])
    ])
    p_ex = ufl.sin(2*pi*x[0])*ufl.cos(2*pi*x[1])
    f_val = ufl.as_vector([
        16*pi**3*ufl.sin(2*pi*x[0])*ufl.cos(2*pi*x[1]) + 2*pi*ufl.cos(2*pi*x[0])*ufl.cos(2*pi*x[1]),
        -16*pi**3*ufl.cos(2*pi*x[0])*ufl.sin(2*pi*x[1]) - 2*pi*ufl.sin(2*pi*x[0])*ufl.sin(2*pi*x[1])
    ])

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f_val, v) * ufl.dx

    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc_u = fem.dirichletbc(u_bc, boundary_dofs, W.sub(0))
    bcs = [bc_u]

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
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

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(msh, tdim)
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

    u_values = np.full((nx_out * ny_out, gdim), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals

    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)

    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    error_u = ufl.inner(u_h - u_ex, u_h - u_ex) * ufl.dx
    error_u_form = fem.form(error_u)
    error_u_local = fem.assemble_scalar(error_u_form)
    error_u_global = np.sqrt(msh.comm.allreduce(error_u_local, op=MPI.SUM))

    p_h = w_h.sub(1).collapse()
    p_h.x.scatter_forward()
    p_ex_func = fem.Function(Q)
    p_ex_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))
    error_p = ufl.inner(p_h - p_ex, p_h - p_ex) * ufl.dx
    error_p_form = fem.form(error_p)
    error_p_local = fem.assemble_scalar(error_p_form)
    error_p_global = np.sqrt(msh.comm.allreduce(error_p_local, op=MPI.SUM))

    if comm.rank == 0:
        print(f"Velocity L2 error: {error_u_global:.6e}")
        print(f"Pressure L2 error: {error_p_global:.6e}")
        print(f"KSP iterations: {iterations}")

    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }
