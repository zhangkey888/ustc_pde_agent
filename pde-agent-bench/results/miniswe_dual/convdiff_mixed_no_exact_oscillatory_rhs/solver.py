import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    if case_spec is None:
        case_spec = {}

    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    epsilon = params.get("epsilon", 0.005)
    beta_vec = params.get("beta", [15.0, 7.0])

    domain_spec = case_spec.get("domain", {})
    x_range = domain_spec.get("x_range", [0.0, 1.0])
    y_range = domain_spec.get("y_range", [0.0, 1.0])

    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)

    # Adaptive mesh refinement with convergence check
    resolutions = [64, 128, 192]
    prev_norm = None
    final_u_grid = None
    final_info = None

    for N in resolutions:
        u_grid, info, l2_norm = _solve_at_resolution(
            N, epsilon, beta_vec, x_range, y_range, nx_out, ny_out
        )
        if prev_norm is not None:
            rel_change = abs(l2_norm - prev_norm) / (abs(l2_norm) + 1e-15)
            if rel_change < 0.01:
                final_u_grid = u_grid
                final_info = info
                break
        prev_norm = l2_norm
        final_u_grid = u_grid
        final_info = info

    return {"u": final_u_grid, "solver_info": final_info}


def _solve_at_resolution(N, epsilon, beta_vec, x_range, y_range, nx_out, ny_out):
    comm = MPI.COMM_WORLD
    p0 = np.array([x_range[0], y_range[0]])
    p1 = np.array([x_range[1], y_range[1]])
    domain = mesh.create_rectangle(
        comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle
    )

    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f_expr = ufl.sin(10 * ufl.pi * x[0]) * ufl.sin(8 * ufl.pi * x[1])
    beta = ufl.as_vector([beta_vec[0], beta_vec[1]])

    # Standard Galerkin
    a_standard = (
        epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    )
    L_standard = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_norm * h / (2.0 * epsilon)
    tau = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_cell) - 1.0 / Pe_cell)

    r_test = ufl.dot(beta, ufl.grad(v))

    # SUPG bilinear: tau * (beta.grad(u)) * (beta.grad(v))
    a_supg = tau * ufl.inner(ufl.dot(beta, ufl.grad(u)), r_test) * ufl.dx
    # SUPG diffusion part: tau * (-eps * div(grad(u))) * (beta.grad(v))
    a_supg_diff = -tau * epsilon * ufl.div(ufl.grad(u)) * r_test * ufl.dx
    # SUPG RHS: tau * f * (beta.grad(v))
    L_supg = tau * f_expr * r_test * ufl.dx

    a_total = a_standard + a_supg + a_supg_diff
    L_total = L_standard + L_supg

    # Boundary conditions: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)

    # Solve with iterative solver, fallback to direct
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10

    try:
        problem = petsc.LinearProblem(
            a_total, L_total, bcs=[bc],
            petsc_options={
                "ksp_type": ksp_type,
                "pc_type": pc_type,
                "ksp_rtol": str(rtol),
                "ksp_max_it": "5000",
                "ksp_gmres_restart": "100",
            },
            petsc_options_prefix="cdiff_"
        )
        u_sol = problem.solve()
    except Exception:
        ksp_type = "preonly"
        pc_type = "lu"
        problem = petsc.LinearProblem(
            a_total, L_total, bcs=[bc],
            petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
            petsc_options_prefix="cdiff_"
        )
        u_sol = problem.solve()

    l2_norm = np.sqrt(
        comm.allreduce(
            fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)),
            op=MPI.SUM
        )
    )

    u_grid = _evaluate_on_grid(domain, u_sol, x_range, y_range, nx_out, ny_out)

    info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 0,
    }

    return u_grid, info, l2_norm


def _evaluate_on_grid(domain, u_func, x_range, y_range, nx, ny):
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    xv, yv = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx * ny))
    points[0, :] = xv.flatten()
    points[1, :] = yv.flatten()

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)

    u_values = np.full(nx * ny, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []

    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = u_func.eval(
            np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()

    return u_values.reshape((nx, ny))


if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {
            "type": "convection_diffusion",
            "parameters": {"epsilon": 0.005, "beta": [15.0, 7.0]},
            "source": "sin(10*pi*x)*sin(8*pi*y)",
        },
        "domain": {"x_range": [0.0, 1.0], "y_range": [0.0, 1.0]},
        "output": {"nx": 50, "ny": 50},
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    u = result["u"]
    print(f"Time: {elapsed:.2f}s")
    print(f"Shape: {u.shape}")
    print(f"Range: [{np.nanmin(u):.8f}, {np.nanmax(u):.8f}]")
    print(f"NaN: {np.sum(np.isnan(u))}")
    print(f"Info: {result['solver_info']}")
