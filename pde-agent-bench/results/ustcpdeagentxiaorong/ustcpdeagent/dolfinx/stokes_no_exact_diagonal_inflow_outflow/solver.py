import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"

from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case_spec
    nu_val = float(case_spec["pde"]["viscosity"])
    f_expr_str = case_spec["pde"]["source_term"]
    bcs_spec = case_spec["pde"]["boundary_conditions"]
    output_spec = case_spec["output"]
    grid = output_spec["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # Mesh resolution - P2/P1 Taylor-Hood on 96x96 mesh gives high accuracy
    N = 96
    degree_u = 2
    degree_p = 1

    # Create mesh
    msh = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [N, N],
        cell_type=mesh.CellType.triangle,
    )
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    # Mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, PETSc.ScalarType((float(f_expr_str[0]), float(f_expr_str[1]))))

    # Stokes bilinear form
    a = (
        nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L_form = ufl.inner(f, v) * ufl.dx

    # Boundary conditions
    bcs = []
    bc_locations = set()

    for bc_item in bcs_spec:
        if bc_item["type"] != "dirichlet":
            continue
        location = bc_item["location"]
        value = bc_item["value"]
        bc_locations.add(location)

        if location == "x0":
            marker = lambda x: np.isclose(x[0], xmin)
        elif location == "x1":
            marker = lambda x: np.isclose(x[0], xmax)
        elif location == "y0":
            marker = lambda x: np.isclose(x[1], ymin)
        elif location == "y1":
            marker = lambda x: np.isclose(x[1], ymax)
        else:
            continue

        facets = mesh.locate_entities_boundary(msh, fdim, marker)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        u_bc_func = fem.Function(V)

        val_strs = list(value)

        def make_interpolator(vs):
            def interpolator(x):
                result = np.zeros((gdim, x.shape[1]))
                for i, expr_str in enumerate(vs):
                    e = expr_str.replace("^", "**")
                    loc = {"x": x[0], "y": x[1], "np": np, "pi": np.pi}
                    try:
                        result[i] = eval(e, {"__builtins__": {}}, loc)
                    except Exception:
                        result[i] = float(e)
                return result
            return interpolator

        u_bc_func.interpolate(make_interpolator(val_strs))
        bc = fem.dirichletbc(u_bc_func, dofs, W.sub(0))
        bcs.append(bc)

    # If all 4 sides have Dirichlet velocity BCs, pin pressure
    all_sides = all(loc in bc_locations for loc in ["x0", "x1", "y0", "y1"])
    if all_sides:
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q),
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
        )
        if len(p_dofs[0]) > 0:
            p0_func = fem.Function(Q)
            p0_func.x.array[:] = 0.0
            bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
            bcs.append(bc_p)

    # Solve with MUMPS direct solver (handles saddle-point systems)
    ksp_type = "preonly"
    pc_type = "lu"

    problem = petsc.LinearProblem(
        a, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_",
    )
    w_h = problem.solve()

    u_h = w_h.sub(0).collapse()

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
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

    u_values = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.reshape(-1, gdim)

    vel_mag = np.sqrt(np.sum(u_values**2, axis=1))
    vel_mag = np.nan_to_num(vel_mag, nan=0.0)
    u_grid = vel_mag.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-10,
        "iterations": 1,
    }

    return {"u": u_grid, "solver_info": solver_info}
