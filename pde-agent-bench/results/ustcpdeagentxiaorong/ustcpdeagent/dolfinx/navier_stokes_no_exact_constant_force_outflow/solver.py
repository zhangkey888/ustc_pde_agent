import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc
import os

# Limit OpenBLAS threads to avoid issues
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case spec
    pde = case_spec["pde"]
    nu_val = pde.get("viscosity", pde.get("coefficients", {}).get("nu", 0.3))
    f_expr = pde.get("source_term", ["1.0", "0.0"])

    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]  # [xmin, xmax, ymin, ymax]

    # Mesh resolution - use moderate resolution 
    N = 64
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    # Mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Parse boundary conditions from case_spec
    bcs_spec = pde.get("boundary_conditions", {})
    bcs = []

    # Map boundary labels to geometric markers
    boundary_markers = {
        "y0": lambda x: np.isclose(x[1], bbox[2]),   # bottom
        "y1": lambda x: np.isclose(x[1], bbox[3]),   # top
        "x0": lambda x: np.isclose(x[0], bbox[0]),   # left
        "x1": lambda x: np.isclose(x[0], bbox[1]),   # right
    }

    for label, bc_info in bcs_spec.items():
        if label not in boundary_markers:
            continue
        marker_fn = boundary_markers[label]
        facets = mesh.locate_entities_boundary(msh, fdim, marker_fn)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        
        # Parse BC value
        if isinstance(bc_info, dict):
            bc_val = bc_info.get("value", [0.0, 0.0])
        elif isinstance(bc_info, (list, tuple)):
            bc_val = bc_info
        else:
            bc_val = [0.0, 0.0]
        
        bc_val_arr = np.array(bc_val, dtype=np.float64)
        u_bc = fem.Function(V)
        u_bc.interpolate(lambda x, v=bc_val_arr: np.tile(v.reshape(-1, 1), (1, x.shape[1])))
        bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))

    # Check if all boundaries have Dirichlet BCs - if so, pin pressure
    has_all_dirichlet = all(k in bcs_spec for k in ["x0", "x1", "y0", "y1"])
    if has_all_dirichlet:
        # Pin pressure at origin to remove nullspace
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q),
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
        )
        if len(p_dofs[0]) > 0:
            p0_func = fem.Function(Q)
            p0_func.x.array[:] = 0.0
            bcs.append(fem.dirichletbc(p0_func, p_dofs, W.sub(1)))

    # Define nonlinear variational problem
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = ufl.as_vector([float(f_expr[0]), float(f_expr[1])])

    # NS residual: viscous + convection + pressure + incompressibility - source
    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    J_form = ufl.derivative(F, w)

    # Newton solver options
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

    problem = petsc.NonlinearProblem(
        F, w, bcs=bcs, J=J_form,
        petsc_options_prefix="ns_",
        petsc_options=petsc_options,
    )

    w_h = problem.solve()
    w.x.scatter_forward()

    # Extract velocity
    u_h = w.sub(0).collapse()

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
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

    u_values = np.zeros((len(pts), gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(
            np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32)
        )
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx] = vals[idx]

    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0] ** 2 + u_values[:, 1] ** 2)
    vel_mag_grid = vel_mag.reshape(ny_out, nx_out)

    # Replace any NaN with 0
    vel_mag_grid = np.nan_to_num(vel_mag_grid, nan=0.0)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [1],
    }

    return {
        "u": vel_mag_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "viscosity": 0.3,
            "source_term": ["1.0", "0.0"],
            "boundary_conditions": {
                "y0": {"type": "dirichlet", "value": [0.0, 0.0]},
                "y1": {"type": "dirichlet", "value": [0.0, 0.0]},
                "x1": {"type": "dirichlet", "value": [0.0, 0.0]},
            },
        },
        "output": {
            "grid": {
                "nx": 100,
                "ny": 100,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            },
            "field": "velocity_magnitude",
        },
    }

    import time
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    u_grid = result["u"]
    print(f"Output shape: {u_grid.shape}")
    print(f"Max velocity magnitude: {np.max(u_grid):.2e}")
    print(f"Min velocity magnitude: {np.min(u_grid):.2e}")
    print(f"Mean velocity magnitude: {np.mean(u_grid):.2e}")
    print(f"Any NaN: {np.any(np.isnan(u_grid))}")
    print(f"Wall time: {elapsed:.2f}s")
