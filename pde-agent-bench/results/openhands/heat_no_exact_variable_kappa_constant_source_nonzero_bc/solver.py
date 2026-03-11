"""
Solver for transient heat equation using dolfinx 0.10.0
Problem: ∂u/∂t - ∇·(κ ∇u) = f in Ω × (0, T]
Case ID: heat_no_exact_variable_kappa_constant_source_nonzero_bc
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _parse_expr(expr_str, x):
    """Convert a string expression to a UFL expression using spatial coordinate x."""
    ns = {
        "x": x[0], "y": x[1],
        "pi": ufl.pi,
        "sin": ufl.sin, "cos": ufl.cos, "exp": ufl.exp,
        "sqrt": ufl.sqrt, "abs": abs,
        "pow": ufl.operators.Power,
    }
    return eval(expr_str, {"__builtins__": {}}, ns)


def _parse_scalar_or_expr(spec, x, domain):
    """Parse a coefficient that may be a constant, expression dict, or raw number."""
    if isinstance(spec, (int, float)):
        return fem.Constant(domain, ScalarType(float(spec)))
    if isinstance(spec, dict):
        if spec.get("type") == "expr":
            return _parse_expr(spec["expr"], x)
        if "value" in spec:
            return fem.Constant(domain, ScalarType(float(spec["value"])))
    return fem.Constant(domain, ScalarType(float(spec)))


def _make_bc_func(bc_spec, V, x_ufl):
    """Create a fem.Function for the boundary condition."""
    u_bc = fem.Function(V)
    if bc_spec is None or bc_spec == 0 or bc_spec == 0.0:
        u_bc.interpolate(lambda x: np.zeros_like(x[0]))
        return u_bc

    if isinstance(bc_spec, (int, float)):
        val = float(bc_spec)
        u_bc.interpolate(lambda x: np.full_like(x[0], val))
        return u_bc

    if isinstance(bc_spec, dict):
        if bc_spec.get("type") == "expr":
            expr_str = bc_spec["expr"]
            expr_ufl = _parse_expr(expr_str, x_ufl)
            expr_compiled = fem.Expression(expr_ufl, V.element.interpolation_points)
            u_bc.interpolate(expr_compiled)
            return u_bc
        if "value" in bc_spec:
            val = float(bc_spec["value"])
            u_bc.interpolate(lambda x: np.full_like(x[0], val))
            return u_bc

    # Fallback: try as float
    val = float(bc_spec)
    u_bc.interpolate(lambda x: np.full_like(x[0], val))
    return u_bc


def solve(case_spec: dict) -> dict:
    # ---- Extract parameters from case_spec ----
    pde = case_spec.get("pde", {})
    time_info = pde.get("time", {})
    coeffs = pde.get("coefficients", {})

    t_end = time_info.get("t_end", case_spec.get("t_end", 0.1))
    dt_suggested = time_info.get("dt", case_spec.get("dt", 0.02))
    scheme = time_info.get("scheme", "backward_euler")

    source_spec = pde.get("source", case_spec.get("source", 1.0))
    kappa_spec = coeffs.get("kappa", case_spec.get("kappa", 1.0))
    bc_spec = pde.get("bc", case_spec.get("bc", 0.0))
    ic_spec = pde.get("initial_condition", case_spec.get("initial_condition", 0.0))

    # ---- Agent-selected parameters ----
    mesh_resolution = 64
    element_degree = 1
    dt = dt_suggested
    ksp_type = "cg"
    pc_type = "ilu"
    rtol = 1e-8

    # ---- Create mesh and function space ----
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                     cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    x = ufl.SpatialCoordinate(domain)

    # ---- Parse kappa ----
    kappa_ufl = _parse_scalar_or_expr(kappa_spec, x, domain)

    # ---- Parse source term ----
    if isinstance(source_spec, (int, float)):
        f_ufl = fem.Constant(domain, ScalarType(float(source_spec)))
    elif isinstance(source_spec, dict) and source_spec.get("type") == "expr":
        f_ufl = _parse_expr(source_spec["expr"], x)
    else:
        f_ufl = fem.Constant(domain, ScalarType(float(source_spec)))

    # ---- Trial / test functions ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)    # previous time step
    u_sol = fem.Function(V)  # current time step

    # ---- Boundary conditions ----
    tdim = domain.topology.dim
    fdim = tdim - 1

    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim,
        lambda x: np.logical_or.reduce([
            np.isclose(x[0], 0.0), np.isclose(x[0], 1.0),
            np.isclose(x[1], 0.0), np.isclose(x[1], 1.0),
        ])
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = _make_bc_func(bc_spec, V, x)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # ---- Initial condition ----
    if isinstance(ic_spec, (int, float)):
        val_ic = float(ic_spec)
        u_n.interpolate(lambda x: np.full_like(x[0], val_ic))
    elif isinstance(ic_spec, dict) and ic_spec.get("type") == "expr":
        ic_expr = fem.Expression(_parse_expr(ic_spec["expr"], x),
                                 V.element.interpolation_points)
        u_n.interpolate(ic_expr)
    else:
        val_ic = float(ic_spec)
        u_n.interpolate(lambda x: np.full_like(x[0], val_ic))

    u_sol.x.array[:] = u_n.x.array[:]

    # Store initial condition for output
    u_n_init = u_n.x.array.copy()

    # ---- Variational forms (backward Euler) ----
    dt_c = fem.Constant(domain, ScalarType(dt))

    a_form = (ufl.inner(u, v) * ufl.dx
              + dt_c * ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L_form = (ufl.inner(u_n, v) * ufl.dx
              + dt_c * ufl.inner(f_ufl, v) * ufl.dx)

    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # Assemble LHS (constant across time steps since kappa doesn't depend on t)
    A = petsc.assemble_matrix(a_compiled, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_compiled)

    # ---- Linear solver ----
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)

    # ---- Time stepping ----
    n_steps = int(np.round(t_end / dt))
    total_iterations = 0

    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()

        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]

    # ---- Sample solution on 50×50 grid ----
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    points_3d = np.vstack([X.ravel(), Y.ravel(), np.zeros(nx_out * ny_out)])

    u_grid = _probe(u_sol, points_3d, domain).reshape(nx_out, ny_out)

    # Initial condition on same grid
    u_init_func = fem.Function(V)
    u_init_func.x.array[:] = u_n_init
    u_initial_grid = _probe(u_init_func, points_3d, domain).reshape(nx_out, ny_out)

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }

    return {"u": u_grid, "u_initial": u_initial_grid, "solver_info": solver_info}


def _probe(u_func, points_array, domain):
    """Evaluate u_func at points_array (shape (3, N)). Returns (N,) array."""
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_array.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_array.T)

    pts, cells, idx_map = [], [], []
    for i in range(points_array.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts.append(points_array.T[i])
            cells.append(links[0])
            idx_map.append(i)

    values = np.full(points_array.shape[1], np.nan)
    if pts:
        vals = u_func.eval(np.array(pts), np.array(cells, dtype=np.int32))
        values[idx_map] = vals.flatten()

    comm = domain.comm
    if comm.size > 1:
        all_vals = comm.gather(values, root=0)
        if comm.rank == 0:
            combined = np.full_like(values, np.nan)
            for pv in all_vals:
                mask = ~np.isnan(pv)
                combined[mask] = pv[mask]
            values = combined
        values = comm.bcast(values, root=0)

    return values