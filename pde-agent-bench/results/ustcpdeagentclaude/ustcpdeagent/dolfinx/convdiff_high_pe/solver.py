import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    eps_val = 0.01
    beta_val = np.array([10.0, 10.0])

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh resolution and element degree
    N = 96
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Exact solution and source
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    # -eps*lap(u) + beta.grad(u) = f
    # lap(u) = -2*pi^2 * sin*sin
    # grad(u) = (pi*cos(pi*x)*sin(pi*y), pi*sin(pi*x)*cos(pi*y))
    lap_u = -2.0 * pi * pi * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    grad_u_x = pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    grad_u_y = pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    f_expr = -eps_val * lap_u + beta_val[0] * grad_u_x + beta_val[1] * grad_u_y

    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    beta = fem.Constant(domain, PETSc.ScalarType(beta_val))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Galerkin part
    a_gal = (eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
             + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx)
    L_gal = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_h = beta_norm * h / (2.0 * eps_c)
    # tau = h / (2 |beta|) * (coth(Pe_h) - 1/Pe_h), approximate with (1 + 1/Pe_h)^-1 form
    # Use standard: tau = h/(2 |beta|) * xi(Pe_h), xi = coth(Pe) - 1/Pe
    # For simplicity use: tau = h / (2|beta|) when Pe >> 1
    tau = h / (2.0 * beta_norm)

    # Residual for SUPG (strong residual of PDE, use div(grad(u)) for TrialFunction with P2)
    # For P2 elements, laplacian term is meaningful
    if degree >= 2:
        R = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f_expr
    else:
        # For P1, drop the diffusive residual part (zero anyway)
        R = ufl.dot(beta, ufl.grad(u)) - f_expr

    # Test function modification (standard SUPG)
    v_supg = tau * ufl.dot(beta, ufl.grad(v))

    a_supg = ufl.inner(R, v_supg) * ufl.dx
    # But R contains f_expr which is RHS - split
    # Actually the SUPG adds: tau * (Lu - f) * (beta.grad(v)) dx
    # We have a_supg above with R = Lu - f, so we need to split properly
    # tau*Lu*v_supg to LHS, tau*f*v_supg to RHS
    if degree >= 2:
        Lu = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    else:
        Lu = ufl.dot(beta, ufl.grad(u))

    a_stab = ufl.inner(Lu, v_supg) * ufl.dx
    L_stab = ufl.inner(f_expr, v_supg) * ufl.dx

    a_form = a_gal + a_stab
    L_form = L_gal + L_stab

    # Dirichlet BC: u = exact on boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a_form, L_form, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()

    its = problem.solver.getIterationNumber()

    # L2 error
    err_form = fem.form((u_sol - u_exact) ** 2 * ufl.dx)
    err_local = fem.assemble_scalar(err_form)
    err = np.sqrt(comm.allreduce(err_local, op=MPI.SUM))

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_vals = np.full(pts.shape[0], np.nan)
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()

    u_grid = u_vals.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "l2_error": float(err),
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    import time
    t0 = time.time()
    out = solve(case_spec)
    t1 = time.time()
    print(f"Time: {t1 - t0:.3f}s")
    print(f"L2 error: {out['l2_error']:.6e}")
    print(f"Shape: {out['u'].shape}")
    # Compute grid error vs exact
    xs = np.linspace(0, 1, 64); ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err_grid = np.sqrt(np.mean((out["u"] - u_ex) ** 2))
    max_err = np.max(np.abs(out["u"] - u_ex))
    print(f"Grid RMS error: {err_grid:.6e}, Max error: {max_err:.6e}")
