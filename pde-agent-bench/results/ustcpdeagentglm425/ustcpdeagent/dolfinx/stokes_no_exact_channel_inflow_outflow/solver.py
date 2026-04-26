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
    pde = case_spec.get("pde", {})
    nu = float(pde.get("viscosity", 1.0))
    out = case_spec.get("output", {})
    grid = out.get("grid", {})
    nx_out = grid.get("nx", 50)
    ny_out = grid.get("ny", 50)
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    mesh_res = 256
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    (u_p, p_p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    a = (2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u_p)), ufl.sym(ufl.grad(v))) * ufl.dx
         - p_p * ufl.div(v) * ufl.dx
         + ufl.div(u_p) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    u_left = fem.Function(V)
    u_left.interpolate(lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[1])]))
    bc_left = fem.dirichletbc(u_left, left_dofs, W.sub(0))
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_bottom = fem.Function(V)
    u_bottom.x.array[:] = 0.0
    bc_bottom = fem.dirichletbc(u_bottom, bottom_dofs, W.sub(0))
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.x.array[:] = 0.0
    bc_top = fem.dirichletbc(u_top, top_dofs, W.sub(0))
    bcs = [bc_left, bc_bottom, bc_top]
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    ksp_type = "gmres"
    pc_type = "lu"
    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 500,
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
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    bb_tree = geometry.bb_tree(msh, tdim)
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
    u_magnitude = np.full(pts.shape[1], np.nan)
    if len(points_on_proc) > 0:
        u_vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        magnitude_vals = np.linalg.norm(u_vals, axis=1)
        u_magnitude[eval_map] = magnitude_vals
    u_grid = u_magnitude.reshape(ny_out, nx_out)
    if comm.size > 1:
        recvbuf = np.zeros_like(u_grid) if comm.rank == 0 else None
        comm.Reduce(u_grid, recvbuf, op=MPI.MAX, root=0)
        if comm.rank == 0:
            u_grid = recvbuf
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }
    return {"u": u_grid, "solver_info": solver_info}
