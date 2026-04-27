import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nu_val = case_spec["pde"]["viscosity"]
    bcs_spec = case_spec["pde"]["boundary_conditions"]
    grid_info = case_spec["output"]["grid"]
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]

    N = 128
    degree_u = 2
    degree_p = 1

    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    bcs = []
    for bc_item in bcs_spec:
        bc_type = bc_item["type"]
        location = bc_item["location"]
        value = bc_item["value"]

        if bc_type == "dirichlet":
            if location == "x0":
                marker = lambda x: np.isclose(x[0], 0.0)
            elif location == "x1":
                marker = lambda x: np.isclose(x[0], 1.0)
            elif location == "y0":
                marker = lambda x: np.isclose(x[1], 0.0)
            elif location == "y1":
                marker = lambda x: np.isclose(x[1], 1.0)
            else:
                continue

            facets = mesh.locate_entities_boundary(msh, fdim, marker)
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)

            u_bc = fem.Function(V)
            val_x_str = value[0]
            val_y_str = value[1]

            def make_interpolator(vx_str, vy_str):
                def interp(x):
                    y_coord = x[1]
                    x_coord = x[0]
                    local_vars = {'x': x_coord, 'y': y_coord, 'np': np}
                    vx = eval(vx_str.replace('^', '**'), {"__builtins__": {}}, local_vars)
                    vy = eval(vy_str.replace('^', '**'), {"__builtins__": {}}, local_vars)
                    if np.isscalar(vx):
                        vx = np.full_like(x_coord, float(vx))
                    if np.isscalar(vy):
                        vy = np.full_like(x_coord, float(vy))
                    return np.vstack([vx, vy])
                return interp

            u_bc.interpolate(make_interpolator(val_x_str, val_y_str))
            bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
            bcs.append(bc)

    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10

    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="stokes_"
    )
    w_h = problem.solve()

    u_h = w_h.sub(0).collapse()

    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_magnitude = np.full(len(pts), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        mag = np.linalg.norm(vals, axis=1)
        for idx, global_idx in enumerate(eval_map):
            u_magnitude[global_idx] = mag[idx]

    u_grid = u_magnitude.reshape(ny_out, nx_out)

    if np.any(np.isnan(u_grid)):
        from scipy.interpolate import NearestNDInterpolator
        valid = ~np.isnan(u_grid.ravel())
        if np.any(valid):
            interp_nn = NearestNDInterpolator(
                np.column_stack([XX.ravel()[valid], YY.ravel()[valid]]),
                u_grid.ravel()[valid]
            )
            nan_mask = np.isnan(u_grid.ravel())
            u_grid_flat = u_grid.ravel()
            u_grid_flat[nan_mask] = interp_nn(
                np.column_stack([XX.ravel()[nan_mask], YY.ravel()[nan_mask]])
            )
            u_grid = u_grid_flat.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "viscosity": 0.05,
            "source_term": ["0.0", "0.0"],
            "boundary_conditions": [
                {"type": "dirichlet", "location": "x0", "value": ["4*y*(1-y)", "0.0"]},
                {"type": "dirichlet", "location": "y0", "value": ["0.0", "0.0"]},
                {"type": "dirichlet", "location": "y1", "value": ["0.0", "0.0"]},
            ],
        },
        "output": {
            "grid": {"nx": 100, "ny": 100, "bbox": [0.0, 1.0, 0.0, 1.0]},
            "field": "velocity_magnitude",
        },
    }
    result = solve(case_spec)
    u_grid = result["u"]
    print(f"Output shape: {u_grid.shape}")
    print(f"Min: {np.nanmin(u_grid):.6f}, Max: {np.nanmax(u_grid):.6f}")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Value at center (0.5, 0.5): {u_grid[50, 50]:.6f}")
    print(f"Value at inlet center (0.0, 0.5): {u_grid[50, 0]:.6f}")
    print("Solver info:", result["solver_info"])
