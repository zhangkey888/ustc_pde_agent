import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract viscosity - handle multiple possible locations
    pde = case_spec["pde"]
    if "viscosity" in pde:
        nu_val = float(pde["viscosity"])
    elif "coefficients" in pde and "nu" in pde["coefficients"]:
        nu_val = float(pde["coefficients"]["nu"])
    else:
        nu_val = 0.22  # default from problem description

    # Extract source term
    f_expr_str = pde.get("source_term", ["0.0", "0.0"])

    # Extract boundary conditions
    bcs_spec = pde.get("boundary_conditions", [])

    # Output grid
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]

    # Mesh resolution
    N = 96
    degree_u = 2
    degree_p = 1

    # Create mesh
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, V_to_W_map = W.sub(0).collapse()
    Q, Q_to_W_map = W.sub(1).collapse()

    # Define boundary conditions
    fdim = msh.topology.dim - 1

    bcs = []

    # Check which boundaries have velocity BCs
    bc_boundaries = set()

    for bc_item in bcs_spec:
        bc_type = bc_item.get("type", "dirichlet")
        bc_value = bc_item.get("value", ["0.0", "0.0"])
        bc_where = bc_item.get("where", "")

        if bc_type == "dirichlet":
            bc_boundaries.add(bc_where)

            # Create marker function
            if bc_where == "x0":
                marker = lambda x: np.isclose(x[0], 0.0)
            elif bc_where == "x1":
                marker = lambda x: np.isclose(x[0], 1.0)
            elif bc_where == "y0":
                marker = lambda x: np.isclose(x[1], 0.0)
            elif bc_where == "y1":
                marker = lambda x: np.isclose(x[1], 1.0)
            else:
                continue

            facets = mesh.locate_entities_boundary(msh, fdim, marker)
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)

            u_bc_func = fem.Function(V)

            if isinstance(bc_value, list) and len(bc_value) == 2:
                val_x_str = str(bc_value[0])
                val_y_str = str(bc_value[1])

                def make_interp(vx_s, vy_s):
                    def interp_func(x):
                        from numpy import sin, cos, pi, exp
                        result = np.zeros((gdim, x.shape[1]))
                        y_coord = x[1]
                        x_coord = x[0]
                        ns = {"sin": sin, "cos": cos, "pi": pi, "exp": exp,
                              "x": x_coord, "y": y_coord, "np": np}
                        vx = eval(vx_s, ns)
                        vy = eval(vy_s, ns)
                        if np.isscalar(vx):
                            result[0, :] = float(vx)
                        else:
                            result[0, :] = vx
                        if np.isscalar(vy):
                            result[1, :] = float(vy)
                        else:
                            result[1, :] = vy
                        return result
                    return interp_func

                u_bc_func.interpolate(make_interp(val_x_str, val_y_str))

            bc = fem.dirichletbc(u_bc_func, dofs, W.sub(0))
            bcs.append(bc)

    # Check if we need pressure pinning
    # If all 4 boundaries have velocity Dirichlet BCs, pressure is determined up to a constant
    all_walls_covered = {"x0", "x1", "y0", "y1"}.issubset(bc_boundaries)
    if all_walls_covered:
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q),
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
        )
        if len(p_dofs[0]) > 0:
            p0_func = fem.Function(Q)
            p0_func.x.array[:] = 0.0
            bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
            bcs.append(bc_p)

    # Define the nonlinear problem
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))

    f_vec = ufl.as_vector([0.0, 0.0])

    # Residual: NS equations
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f_vec, v) * ufl.dx
    )

    J_form = ufl.derivative(F, w)

    # Initialize with zero
    w.x.array[:] = 0.0

    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J_form,
                                      petsc_options_prefix="ns_",
                                      petsc_options=petsc_options)

    w_h = problem.solve()
    w.x.scatter_forward()

    # Extract velocity
    u_h = w.sub(0).collapse()

    # Sample onto output grid
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
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx_local, idx_global in enumerate(eval_map):
            u_values[idx_global, :] = vals[idx_local, :]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    vel_mag = np.nan_to_num(vel_mag, nan=0.0)

    u_grid = vel_mag.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [15],
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time

    case_spec = {
        "pde": {
            "viscosity": 0.22,
            "source_term": ["0.0", "0.0"],
            "boundary_conditions": [
                {"type": "dirichlet", "value": ["sin(pi*y)", "0.2*sin(2*pi*y)"], "where": "x0"},
                {"type": "dirichlet", "value": ["0.0", "0.0"], "where": "y0"},
                {"type": "dirichlet", "value": ["0.0", "0.0"], "where": "y1"},
            ],
        },
        "output": {
            "field": "velocity_magnitude",
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
        },
    }

    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start

    print(f"Wall time: {elapsed:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Min velocity magnitude: {np.min(result['u']):.10f}")
    print(f"Max velocity magnitude: {np.max(result['u']):.10f}")
    print(f"Mean velocity magnitude: {np.mean(result['u']):.10f}")
    print(f"Any NaN: {np.any(np.isnan(result['u']))}")

    u = result['u']
    print(f"Bottom row (y=0): min={np.min(u[0,:]):.8f}, max={np.max(u[0,:]):.8f}")
    print(f"Top row (y=1): min={np.min(u[-1,:]):.8f}, max={np.max(u[-1,:]):.8f}")
    print(f"Left col (x=0): min={np.min(u[:,0]):.8f}, max={np.max(u[:,0]):.8f}")
    print(f"Right col (x=1): min={np.min(u[:,-1]):.8f}, max={np.max(u[:,-1]):.8f}")
    print("SUCCESS")
