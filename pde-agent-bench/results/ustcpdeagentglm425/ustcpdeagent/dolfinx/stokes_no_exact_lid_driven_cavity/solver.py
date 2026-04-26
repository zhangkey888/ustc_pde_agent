import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    gdim = 2

    nu_val = float(case_spec["pde"]["parameters"]["viscosity"]["value"])
    f_vals = case_spec["pde"].get("source", ["0.0", "0.0"])
    f_vec = [float(v) for v in f_vals]

    bcs_spec = case_spec["pde"].get("boundary_conditions", {})
    dirichlet = bcs_spec.get("dirichlet", {})

    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]

    mesh_res = 256

    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    f_const = fem.Constant(msh, PETSc.ScalarType(f_vec))

    a = (2.0 * nu_val * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f_const, v) * ufl.dx

    fdim = msh.topology.dim - 1
    bcs = []

    def make_geom_func(name):
        if name == "y1":
            return lambda x: np.isclose(x[1], 1.0)
        elif name == "y0":
            return lambda x: np.isclose(x[1], 0.0)
        elif name == "x0":
            return lambda x: np.isclose(x[0], 0.0)
        elif name == "x1":
            return lambda x: np.isclose(x[0], 1.0)
        return None

    for bc_name, bc_val in dirichlet.items():
        geom_func = make_geom_func(bc_name)
        if geom_func is None:
            continue
        facets = mesh.locate_entities_boundary(msh, fdim, geom_func)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        u_bc = fem.Function(V)
        v0, v1 = float(bc_val[0]), float(bc_val[1])
        u_bc.interpolate(lambda X: np.vstack([
            np.full(X.shape[1], v0),
            np.full(X.shape[1], v1)
        ]))
        bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0_func, p_dofs, W.sub(1)))

    ksp_type = "preonly"
    pc_type = "lu"

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

    u_h = w_h.sub(0).collapse()

    iterations = problem.solver.getIterationNumber()

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
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

    u_grid = np.zeros((ny_out, nx_out))
    if len(points_on_proc) > 0:
        vals = u_h.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        magnitudes = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        u_flat = np.zeros(nx_out * ny_out)
        u_flat[eval_map] = magnitudes
        u_grid = u_flat.reshape(ny_out, nx_out)

    u_grid_global = np.zeros_like(u_grid)
    comm.Allreduce(u_grid, u_grid_global, op=MPI.SUM)

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-10,
        "iterations": iterations,
    }

    return {"u": u_grid_global, "solver_info": solver_info}
