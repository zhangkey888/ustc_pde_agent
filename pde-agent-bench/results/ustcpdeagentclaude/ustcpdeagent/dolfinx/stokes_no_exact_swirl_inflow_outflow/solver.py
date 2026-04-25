import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    nu_val = 0.5
    N = 256  # mesh resolution
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    deg_u, deg_p = 2, 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1

    # x = 0 inflow
    def on_x0(x): return np.isclose(x[0], 0.0)
    def on_y0(x): return np.isclose(x[1], 0.0)
    def on_y1(x): return np.isclose(x[1], 1.0)

    facets_x0 = mesh.locate_entities_boundary(msh, fdim, on_x0)
    facets_y0 = mesh.locate_entities_boundary(msh, fdim, on_y0)
    facets_y1 = mesh.locate_entities_boundary(msh, fdim, on_y1)

    u_x0 = fem.Function(V)
    u_x0.interpolate(lambda x: np.vstack([np.sin(np.pi * x[1]),
                                           0.2 * np.sin(2 * np.pi * x[1])]))
    dofs_x0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x0)
    bc_x0 = fem.dirichletbc(u_x0, dofs_x0, W.sub(0))

    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y0)
    bc_y0 = fem.dirichletbc(u_zero, dofs_y0, W.sub(0))
    dofs_y1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y1)
    bc_y1 = fem.dirichletbc(u_zero, dofs_y1, W.sub(0))

    bcs = [bc_x0, bc_y0, bc_y1]
    # x=1 is outflow: natural BC (do-nothing) — this fixes pressure gauge, no pin needed.

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = fem_petsc.LinearProblem(a, L, bcs=bcs,
                                      petsc_options=petsc_options,
                                      petsc_options_prefix="stokes_")
    w_h = problem.solve()

    # Get solver iteration count
    try:
        iters = problem.solver.getIterationNumber()
    except Exception:
        iters = 1

    u_h = w_h.sub(0).collapse()

    # Sample velocity magnitude on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(XX.size)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    u_values = np.zeros((pts.shape[0], 2))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[idx_map] = vals

    mag = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)

    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(iters),
        }
    }


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    print(f"Time: {time.time() - t0:.3f}s")
    print(f"Shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {result['u'].max():.6f}")
    print(f"Min velocity magnitude: {result['u'].min():.6f}")
    print(f"Solver info: {result['solver_info']}")
