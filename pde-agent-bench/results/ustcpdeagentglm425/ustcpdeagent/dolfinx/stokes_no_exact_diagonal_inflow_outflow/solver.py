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
    pde = case_spec["pde"]
    nu_val = float(pde["viscosity"])
    output_spec = case_spec["output"]
    grid_spec = output_spec["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]

    mesh_res = 64
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, np.zeros(gdim, dtype=PETSc.ScalarType))

    def eps(u):
        return ufl.sym(ufl.grad(u))

    a = (2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    bcs = []

    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    u_inflow = fem.Function(V)
    u_inflow.interpolate(lambda x: np.array([2.0*x[1]*(1.0-x[1]), 2.0*x[1]*(1.0-x[1])]))
    bcs.append(fem.dirichletbc(u_inflow, left_dofs, W.sub(0)))

    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_bottom = fem.Function(V)
    u_bottom.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_bottom, bottom_dofs, W.sub(0)))

    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_top, top_dofs, W.sub(0)))

    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-8

    petsc_options = {
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "ksp_rtol": str(rtol),
        "ksp_atol": "1e-10",
        "ksp_max_it": "5000",
    }

    problem = petsc.LinearProblem(a, L, bcs=bcs,
                                   petsc_options=petsc_options,
                                   petsc_options_prefix="stokes_")
    w_h = problem.solve()
    w_h.x.scatter_forward()

    iterations = problem.solver.getIterationNumber()

    u_h = w_h.sub(0).collapse()
    u_h.x.scatter_forward()

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)

    pts = np.zeros((nx_out * ny_out, 3))
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

    u_values = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals

    magnitude = np.sqrt(np.nansum(u_values**2, axis=1)).reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }

    if "time" in pde:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 1
        solver_info["time_scheme"] = "steady"

    return {
        "u": magnitude,
        "solver_info": solver_info,
    }
