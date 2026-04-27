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
    nu_val = float(case_spec["pde"]["coefficients"]["nu"])
    f_expr = case_spec["pde"]["source_term"]
    bcs_spec = case_spec["pde"]["boundary_conditions"]

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # Mesh parameters
    N = 128
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

    # Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Variational problem
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    nu_c = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f_vec = fem.Constant(
        msh, np.array([float(f_expr[0]), float(f_expr[1])], dtype=PETSc.ScalarType)
    )

    a_form = (
        nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        - p * ufl.div(v) * ufl.dx
        + ufl.div(u) * q * ufl.dx
    )
    L_form = ufl.inner(f_vec, v) * ufl.dx

    # --- Boundary conditions ---
    bcs = []

    for bc_item in bcs_spec:
        location = bc_item["location"]
        value = bc_item["value"]
        v0 = float(value[0])
        v1 = float(value[1])

        # Define marker - use explicit if/elif to avoid closure issues
        if location == "y1":
            marker_fn = lambda x: np.isclose(x[1], ymax)
        elif location == "y0":
            marker_fn = lambda x: np.isclose(x[1], ymin)
        elif location == "x0":
            marker_fn = lambda x: np.isclose(x[0], xmin)
        elif location == "x1":
            marker_fn = lambda x: np.isclose(x[0], xmax)
        else:
            raise ValueError(f"Unknown location: {location}")

        facets = mesh.locate_entities_boundary(msh, fdim, marker_fn)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)

        u_bc = fem.Function(V)
        if abs(v0) < 1e-15 and abs(v1) < 1e-15:
            u_bc.x.array[:] = 0.0
        else:
            u_bc.interpolate(
                lambda x, _v0=v0, _v1=v1: np.vstack(
                    [np.full(x.shape[1], _v0), np.full(x.shape[1], _v1)]
                )
            )

        bcs.append(fem.dirichletbc(u_bc, dofs, W.sub(0)))

    # Pressure pin at corner
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], xmin) & np.isclose(x[1], ymin),
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    # --- Solve with MUMPS ---
    w_h = petsc.LinearProblem(
        a_form,
        L_form,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_",
    ).solve()

    # Extract velocity
    u_h = w_h.sub(0).collapse()

    # --- Sample on output grid ---
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

    u_values = np.zeros((len(pts), gdim))
    if len(points_on_proc) > 0:
        vals = u_h.eval(
            np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32)
        )
        if vals.ndim == 1:
            vals = vals.reshape(-1, gdim)
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]

    vel_mag = np.sqrt(u_values[:, 0] ** 2 + u_values[:, 1] ** 2)
    u_grid = vel_mag.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-12,
        "iterations": 1,
    }

    return {"u": u_grid, "solver_info": solver_info}
