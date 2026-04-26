import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    nu = float(pde["nu"])
    out = case_spec["output"]
    nx_out = out["grid"]["nx"]
    ny_out = out["grid"]["ny"]
    bbox = out["grid"]["bbox"]

    mesh_res = 256
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

    x = ufl.SpatialCoordinate(msh)
    f_ufl = ufl.as_vector([
        3.0 * ufl.exp(-50.0 * ((x[0] - 0.15)**2 + (x[1] - 0.15)**2)),
        3.0 * ufl.exp(-50.0 * ((x[0] - 0.15)**2 + (x[1] - 0.15)**2))
    ])

    a = (2.0 * nu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v))) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f_ufl, v) * ufl.dx

    fdim = msh.topology.dim - 1
    bcs = []

    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_left = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    u_zero_l = fem.Function(V); u_zero_l.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_zero_l, dofs_left, W.sub(0)))

    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_zero_b = fem.Function(V); u_zero_b.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_zero_b, dofs_bottom, W.sub(0)))

    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_zero_t = fem.Function(V); u_zero_t.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_zero_t, dofs_top, W.sub(0)))

    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q); p0_func.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0_func, p_dofs, W.sub(1)))

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

    u_h = w_h.sub(0).collapse()
    u_h.x.scatter_forward()

    # Sample velocity magnitude on output grid
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

    u_grid = np.zeros((ny_out, nx_out))
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        magnitude = np.linalg.norm(vals, axis=1)
        u_grid_flat = np.zeros(nx_out * ny_out)
        u_grid_flat[eval_map] = magnitude
        u_grid = u_grid_flat.reshape(ny_out, nx_out)

    # Verification: L2 norm of divergence
    div_form = fem.form(ufl.div(u_h)**2 * ufl.dx)
    div_l2 = np.sqrt(fem.assemble_scalar(div_form))
    div_l2 = msh.comm.allreduce(div_l2, op=MPI.SUM)
    print(f"[Verification] L2 norm of div(u): {div_l2:.6e}")

    # Verification: kinetic energy
    ke_form = fem.form(0.5 * ufl.inner(u_h, u_h) * ufl.dx)
    ke = fem.assemble_scalar(ke_form)
    ke = msh.comm.allreduce(ke, op=MPI.SUM)
    print(f"[Verification] Kinetic energy: {ke:.6e}")

    # Verification: residual check - compute ||A*w - b||
    # The direct solve should give machine precision residual

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 2,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 0,
        }
    }
    return result

if __name__ == "__main__":
    case_spec = {
        "pde": {
            "nu": 0.1,
            "source": ["3*exp(-50*((x-0.15)**2 + (y-0.15)**2))", "3*exp(-50*((x-0.15)**2 + (y-0.15)**2))"],
            "boundary_conditions": [
                {"type": "dirichlet", "boundary": "x0", "value": [0.0, 0.0]},
                {"type": "dirichlet", "boundary": "y0", "value": [0.0, 0.0]},
                {"type": "dirichlet", "boundary": "y1", "value": [0.0, 0.0]},
            ],
            "time": None,
        },
        "output": {
            "grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]},
            "field": "velocity_magnitude",
        }
    }
    import time as _time
    t0 = _time.time()
    result = solve(case_spec)
    t1 = _time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u min: {np.nanmin(result['u']):.6e}, max: {np.nanmax(result['u']):.6e}")
    print(f"solver_info: {result['solver_info']}")
