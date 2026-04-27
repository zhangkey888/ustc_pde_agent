import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case_spec
    pde = case_spec["pde"]

    if "viscosity" in pde:
        nu_val = float(pde["viscosity"])
    elif "coefficients" in pde and "nu" in pde["coefficients"]:
        nu_val = float(pde["coefficients"]["nu"])
    else:
        nu_val = 0.08

    if "source_term" in pde:
        f_expr = pde["source_term"]
    elif "source" in pde:
        f_expr = pde["source"]
    else:
        f_expr = ["0.0", "0.0"]

    if "boundary_conditions" in pde:
        bcs_spec = pde["boundary_conditions"]
    elif "bcs" in pde:
        bcs_spec = pde["bcs"]
    else:
        bcs_spec = []

    output_spec = case_spec["output"]
    nx_out = output_spec["grid"]["nx"]
    ny_out = output_spec["grid"]["ny"]
    bbox = output_spec["grid"]["bbox"]

    # Mesh resolution - use fine mesh for accuracy within time budget
    N = 256
    degree_u = 2
    degree_p = 1

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    bcs = []

    for bc_item in bcs_spec:
        bc_type = bc_item.get("type", "dirichlet")
        location = bc_item.get("location", "")
        value = bc_item.get("value", [0.0, 0.0])

        if bc_type == "dirichlet":
            if location == "y1":
                marker = lambda x: np.isclose(x[1], 1.0)
            elif location == "y0":
                marker = lambda x: np.isclose(x[1], 0.0)
            elif location == "x0":
                marker = lambda x: np.isclose(x[0], 0.0)
            elif location == "x1":
                marker = lambda x: np.isclose(x[0], 1.0)
            else:
                continue

            facets = mesh.locate_entities_boundary(msh, fdim, marker)
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)

            u_bc = fem.Function(V)
            val = np.array(value, dtype=np.float64)
            u_bc.interpolate(lambda x, v=val: np.outer(v, np.ones(x.shape[1])))

            bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
            bcs.append(bc)

    # Pressure pinning at (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs[0]) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f_val = np.array([float(f_expr[0]), float(f_expr[1])], dtype=np.float64)
    f = fem.Constant(msh, PETSc.ScalarType(f_val))

    F = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + q * ufl.div(u) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    J_form = ufl.derivative(F, w)

    w.x.array[:] = 0.0

    petsc_opts = {
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
        petsc_options=petsc_opts
    )

    problem.solve()
    w.x.scatter_forward()

    u_h = w.sub(0).collapse()

    # Sample onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()

    tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
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

    u_grid = np.zeros(nx_out * ny_out)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        magnitudes = np.linalg.norm(vals, axis=1)
        for idx, i in enumerate(eval_map):
            u_grid[i] = magnitudes[idx]

    u_grid = u_grid.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "nonlinear_iterations": [3],
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "viscosity": 0.08,
            "source_term": ["0.0", "0.0"],
            "boundary_conditions": [
                {"type": "dirichlet", "location": "y1", "value": [1.0, 0.0]},
                {"type": "dirichlet", "location": "y0", "value": [0.0, 0.0]},
                {"type": "dirichlet", "location": "x0", "value": [0.0, 0.0]},
                {"type": "dirichlet", "location": "x1", "value": [0.0, 0.0]},
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
    t1 = time.time()
    print(f"Solve time: {t1 - t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max velocity magnitude: {np.max(result['u']):.6f}")
    print(f"Min velocity magnitude: {np.min(result['u']):.6f}")
    print(f"NaN count: {np.sum(np.isnan(result['u']))}")
    print(f"Top row (y=1): min={np.min(result['u'][-1,:]):.6f}, max={np.max(result['u'][-1,:]):.6f}")
    print(f"Center value: {result['u'][50, 50]:.6f}")
