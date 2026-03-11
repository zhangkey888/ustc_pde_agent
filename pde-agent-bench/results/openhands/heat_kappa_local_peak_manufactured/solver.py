"""
Solver for transient heat equation using dolfinx 0.10.0
Problem: ∂u/∂t - ∇·(κ ∇u) = f in Ω × (0, T]
Case ID: heat_kappa_local_peak_manufactured
Manufactured solution: u = exp(-t)*sin(πx)*sin(2πy)
"""

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

ScalarType = PETSc.ScalarType


def _parse_expr(expr_str, x, extra_ns=None):
    """Convert a string expression to a UFL expression using spatial coordinate x."""
    ns = {
        "x": x[0], "y": x[1],
        "pi": ufl.pi,
        "sin": ufl.sin, "cos": ufl.cos, "exp": ufl.exp,
        "sqrt": ufl.sqrt, "abs": abs,
        "pow": ufl.operators.Power,
    }
    if extra_ns:
        ns.update(extra_ns)
    return eval(expr_str, {"__builtins__": {}}, ns)


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    time_info = pde.get("time", {})
    t_end = time_info.get("t_end", 0.1)

    # ---- Agent-selected parameters ----
    mesh_resolution = 64
    element_degree = 2
    dt = 0.002
    n_steps = int(round(t_end / dt))
    ksp_type = "cg"
    pc_type = "ilu"
    rtol = 1e-10

    # ---- Create mesh and function space ----
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution,
                                     cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    xc = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # ---- Time parameter (updated each step) ----
    t_param = fem.Constant(domain, ScalarType(0.0))

    # ---- Kappa: 1 + 30*exp(-150*((x-0.35)^2 + (y-0.65)^2)) ----
    g_peak = ufl.exp(-150.0 * ((xc[0] - 0.35)**2 + (xc[1] - 0.65)**2))
    kappa = 1.0 + 30.0 * g_peak

    # ---- Manufactured solution: u = exp(-t)*sin(πx)*sin(2πy) ----
    # Analytically derived source term f = ∂u/∂t - ∇·(κ∇u)
    # ∂u/∂t = -exp(-t)*sin(πx)*sin(2πy)
    # ∇·(κ∇u) = (∂κ/∂x)(∂u/∂x) + (∂κ/∂y)(∂u/∂y) + κ*(∂²u/∂x² + ∂²u/∂y²)
    # ∂κ/∂x = -9000*(x-0.35)*g, ∂κ/∂y = -9000*(y-0.65)*g
    # ∂u/∂x = exp(-t)*π*cos(πx)*sin(2πy)
    # ∂u/∂y = exp(-t)*sin(πx)*2π*cos(2πy)
    # Δu = -5π²*exp(-t)*sin(πx)*sin(2πy)
    # f = exp(-t)*[(5π²κ - 1)*sin(πx)*sin(2πy)
    #              - ∂κ/∂x * π*cos(πx)*sin(2πy)
    #              - ∂κ/∂y * 2π*sin(πx)*cos(2πy)]

    sx = ufl.sin(pi * xc[0])
    sy = ufl.sin(2 * pi * xc[1])
    cx = ufl.cos(pi * xc[0])
    cy = ufl.cos(2 * pi * xc[1])
    exp_t = ufl.exp(-t_param)

    dkappa_dx = -9000.0 * (xc[0] - 0.35) * g_peak
    dkappa_dy = -9000.0 * (xc[1] - 0.65) * g_peak

    f_ufl = exp_t * (
        (5.0 * pi**2 * kappa - 1.0) * sx * sy
        - dkappa_dx * pi * cx * sy
        - dkappa_dy * 2.0 * pi * sx * cy
    )

    # ---- Trial / test functions ----
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)   # previous time step
    u_sol = fem.Function(V) # solution at current step

    # ---- Boundary conditions (homogeneous: sin(πx)=0 at x=0,1; sin(2πy)=0 at y=0,1) ----
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
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    # ---- Initial condition: u(x,0) = sin(πx)*sin(2πy) ----
    u_exact_t0 = sx * sy
    ic_expr = fem.Expression(u_exact_t0, V.element.interpolation_points)
    u_n.interpolate(ic_expr)
    u_sol.x.array[:] = u_n.x.array[:]

    # Store initial condition for output
    u_n_init = u_n.x.array.copy()

    # ---- Variational forms (backward Euler) ----
    # (u^{n+1} - u^n)/dt - ∇·(κ∇u^{n+1}) = f^{n+1}
    # => ∫ u v dx + dt ∫ κ ∇u·∇v dx = ∫ u_n v dx + dt ∫ f v dx
    dt_c = fem.Constant(domain, ScalarType(dt))

    a_form = (ufl.inner(u, v) * ufl.dx
              + dt_c * ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx)
    L_form = (ufl.inner(u_n, v) * ufl.dx
              + dt_c * ufl.inner(f_ufl, v) * ufl.dx)

    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    # Assemble LHS once (kappa is time-independent)
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
    total_iterations = 0

    for step in range(n_steps):
        t_param.value = (step + 1) * dt

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