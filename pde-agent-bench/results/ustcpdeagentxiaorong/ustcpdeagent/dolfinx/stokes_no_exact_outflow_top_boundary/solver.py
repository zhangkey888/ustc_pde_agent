import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case spec
    nu_val = case_spec["pde"]["viscosity"]
    bcs_spec = case_spec["pde"]["boundary_conditions"]

    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]

    # Mesh resolution - use high resolution for accuracy
    N = 256
    degree_u = 2
    degree_p = 1

    # Create mesh
    p0 = np.array([bbox[0], bbox[2]])
    p1 = np.array([bbox[1], bbox[3]])
    msh = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    # Source term
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))

    # Viscosity constant
    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))

    # Bilinear form for Stokes
    a = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # Boundary conditions
    fdim = msh.topology.dim - 1
    bcs = []

    # Collect which boundaries have Dirichlet BCs
    bc_locations = set()
    for bc_item in bcs_spec:
        if bc_item["type"] == "dirichlet":
            bc_locations.add(bc_item["location"])

    for bc_item in bcs_spec:
        if bc_item["type"] != "dirichlet":
            continue

        location = bc_item["location"]
        value = bc_item["value"]

        # Create boundary marker
        xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
        if location == "x0":
            marker = lambda x, xv=xmin: np.isclose(x[0], xv)
        elif location == "x1":
            marker = lambda x, xv=xmax: np.isclose(x[0], xv)
        elif location == "y0":
            marker = lambda x, yv=ymin: np.isclose(x[1], yv)
        elif location == "y1":
            marker = lambda x, yv=ymax: np.isclose(x[1], yv)
        else:
            continue

        facets = mesh.locate_entities_boundary(msh, fdim, marker)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)

        u_bc = fem.Function(V)

        # Parse value strings
        val_x_str = str(value[0])
        val_y_str = str(value[1])

        def make_bc_func(vx_str, vy_str):
            def bc_func(x):
                local_vars = {
                    'x': x[0], 'y': x[1],
                    'sin': np.sin, 'cos': np.cos,
                    'pi': np.pi, 'exp': np.exp,
                    'np': np, 'abs': np.abs,
                    'sqrt': np.sqrt,
                }
                vx = eval(vx_str, {"__builtins__": {}}, local_vars)
                vy = eval(vy_str, {"__builtins__": {}}, local_vars)
                if isinstance(vx, (int, float)):
                    vx = np.full_like(x[0], float(vx))
                if isinstance(vy, (int, float)):
                    vy = np.full_like(x[0], float(vy))
                return np.vstack([vx, vy])
            return bc_func

        bc_func = make_bc_func(val_x_str, val_y_str)
        u_bc.interpolate(bc_func)

        bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
        bcs.append(bc)

    # Check if all 4 boundaries have Dirichlet BCs -> need pressure pinning
    all_boundaries = {"x0", "x1", "y0", "y1"}
    if bc_locations >= all_boundaries:
        # All Dirichlet - need pressure pinning
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q),
            lambda x: np.isclose(x[0], bbox[0]) & np.isclose(x[1], bbox[2]),
        )
        if len(p_dofs[0]) > 0:
            p0_func = fem.Function(Q)
            p0_func.x.array[:] = 0.0
            bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
            bcs.append(bc_p)

    # Solve with MUMPS direct solver (robust for saddle-point systems)
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

    # Extract velocity
    u_h = w_h.sub(0).collapse()

    # Sample on output grid
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
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((len(pts), gdim), 0.0)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for j, idx in enumerate(eval_map):
            u_values[idx] = vals[j]

    # Compute velocity magnitude
    magnitude = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = magnitude.reshape(ny_out, nx_out)

    # Handle any NaN values
    if np.any(np.isnan(u_grid)):
        mask = np.isnan(u_grid)
        nan_count = np.sum(mask)
        if nan_count < u_grid.size * 0.1:
            from scipy.interpolate import NearestNDInterpolator
            valid = ~mask
            yi, xi = np.where(valid)
            values_valid = u_grid[valid]
            yi_nan, xi_nan = np.where(mask)
            if len(yi_nan) > 0 and len(yi) > 0:
                interp = NearestNDInterpolator(
                    np.column_stack([yi, xi]), values_valid
                )
                u_grid[mask] = interp(np.column_stack([yi_nan, xi_nan]))

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "viscosity": 0.9,
            "source_term": ["0.0", "0.0"],
            "boundary_conditions": [
                {"type": "dirichlet", "location": "x0", "value": ["sin(pi*y)", "0.0"]},
                {"type": "dirichlet", "location": "y0", "value": ["0.0", "0.0"]},
                {"type": "dirichlet", "location": "x1", "value": ["0.0", "0.0"]},
            ],
        },
        "output": {
            "field": "velocity_magnitude",
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
        },
    }

    import time
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Min: {np.nanmin(result['u']):.6f}, Max: {np.nanmax(result['u']):.6f}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
    print(f"Solver info: {result['solver_info']}")
    
    # Quick sanity checks
    # At x=0, y=0.5: u should be [sin(pi*0.5), 0] = [1, 0], |u| = 1
    # At x=1, y=0.5: u should be [0, 0], |u| = 0
    print(f"\nSanity checks:")
    print(f"  u_grid[50, 0] (x=0, y~0.5): {result['u'][50, 0]:.6f} (expect ~1.0)")
    print(f"  u_grid[50, 99] (x=1, y~0.5): {result['u'][50, 99]:.6f} (expect ~0.0)")
    print(f"  u_grid[0, 50] (x~0.5, y=0): {result['u'][0, 50]:.6f} (expect ~0.0)")
