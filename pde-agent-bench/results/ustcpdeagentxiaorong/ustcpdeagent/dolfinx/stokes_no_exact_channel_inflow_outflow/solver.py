import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    nu_val = case_spec["pde"]["viscosity"]
    bcs_spec = case_spec["pde"]["bcs"]

    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    N = 128
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, V_map = W.sub(0).collapse()
    Q, Q_map = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    fdim = msh.topology.dim - 1
    bcs = []

    for bc_entry in bcs_spec:
        bc_type = bc_entry["type"]
        bc_value = bc_entry["value"]
        bc_loc = bc_entry["location"]

        if bc_type == "dirichlet":
            if bc_loc == "x0":
                marker = lambda x: np.isclose(x[0], 0.0)
            elif bc_loc == "x1":
                marker = lambda x: np.isclose(x[0], 1.0)
            elif bc_loc == "y0":
                marker = lambda x: np.isclose(x[1], 0.0)
            elif bc_loc == "y1":
                marker = lambda x: np.isclose(x[1], 1.0)
            else:
                raise ValueError(f"Unknown location: {bc_loc}")

            facets = mesh.locate_entities_boundary(msh, fdim, marker)
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)

            u_bc = fem.Function(V)
            val_x_str = bc_value[0]
            val_y_str = bc_value[1]

            def make_bc_func(vx_str, vy_str):
                def bc_func(x):
                    y_coord = x[1]
                    x_coord = x[0]
                    safe_ns = {"x": x_coord, "y": y_coord, "np": np,
                               "pi": np.pi, "sin": np.sin, "cos": np.cos,
                               "exp": np.exp, "abs": np.abs}
                    vx = eval(vx_str, {"__builtins__": {}}, safe_ns)
                    vy = eval(vy_str, {"__builtins__": {}}, safe_ns)
                    if np.isscalar(vx):
                        vx = np.full_like(x[0], float(vx))
                    if np.isscalar(vy):
                        vy = np.full_like(x[0], float(vy))
                    return np.vstack([vx, vy])
                return bc_func

            bc_func = make_bc_func(val_x_str, val_y_str)
            u_bc.interpolate(bc_func)
            bc_obj = fem.dirichletbc(u_bc, dofs, W.sub(0))
            bcs.append(bc_obj)

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
    u_h = w_h.sub(0).collapse()

    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_indices = []

    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_indices.append(i)

    u_grid = np.full((ny_out * nx_out,), np.nan)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        u_vals = u_h.eval(pts_arr, cells_arr)
        vel_mag = np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)
        for idx, gi in enumerate(eval_indices):
            u_grid[gi] = vel_mag[idx]

    u_grid = u_grid.reshape(ny_out, nx_out)

    if np.any(np.isnan(u_grid)):
        from scipy.interpolate import NearestNDInterpolator
        valid = ~np.isnan(u_grid.ravel())
        if np.any(valid):
            interp = NearestNDInterpolator(
                np.c_[XX.ravel()[valid], YY.ravel()[valid]],
                u_grid.ravel()[valid]
            )
            nan_mask = np.isnan(u_grid.ravel())
            filled = u_grid.ravel().copy()
            filled[nan_mask] = interp(
                np.c_[XX.ravel()[nan_mask], YY.ravel()[nan_mask]]
            )
            u_grid = filled.reshape(ny_out, nx_out)

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 1,
        }
    }

    return result


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "viscosity": 1.0,
            "source": ["0.0", "0.0"],
            "bcs": [
                {"type": "dirichlet", "value": ["4*y*(1-y)", "0.0"], "location": "x0"},
                {"type": "dirichlet", "value": ["0.0", "0.0"], "location": "y0"},
                {"type": "dirichlet", "value": ["0.0", "0.0"], "location": "y1"},
            ]
        },
        "output": {
            "field": "velocity_magnitude",
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }

    import time
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Min velocity magnitude: {np.min(u_grid):.6f}")
    print(f"Max velocity magnitude: {np.max(u_grid):.6f}")
    print(f"Mean velocity magnitude: {np.mean(u_grid):.6f}")
    print(f"NaN count: {np.sum(np.isnan(u_grid))}")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Solver info: {result['solver_info']}")
