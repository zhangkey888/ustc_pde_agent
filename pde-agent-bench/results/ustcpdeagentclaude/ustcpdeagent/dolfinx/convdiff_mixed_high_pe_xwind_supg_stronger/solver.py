import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    eps_val = 0.005
    beta_val = np.array([20.0, 0.0])

    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Mesh - high resolution for high Pe
    N = 160
    degree = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(msh, ("Lagrange", degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # f = -eps * Lap(u) + beta . grad(u)
    # Lap u = -2 pi^2 sin(pi x) sin(pi y)
    # grad u = (pi cos(pi x) sin(pi y), pi sin(pi x) cos(pi y))
    f_expr = (
        -eps_val * (-2 * pi * pi * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]))
        + beta_val[0] * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
        + beta_val[1] * pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])
    )

    eps_c = fem.Constant(msh, PETSc.ScalarType(eps_val))
    beta_c = fem.Constant(msh, PETSc.ScalarType(beta_val))

    # Standard Galerkin
    a = eps_c * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.dot(beta_c, ufl.grad(u)) * v * ufl.dx
    L = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(msh)
    beta_norm = ufl.sqrt(ufl.dot(beta_c, beta_c) + 1e-14)
    Pe_h = beta_norm * h / (2.0 * eps_c)
    # tau - for high Pe: h/(2|beta|) * (coth(Pe) - 1/Pe) ~ h/(2|beta|)
    # Use standard formula
    tau = (h / (2.0 * beta_norm)) * (1.0 / ufl.tanh(Pe_h) - 1.0 / Pe_h)

    # SUPG residual: R(u) = -eps Lap(u) + beta.grad(u) - f
    # For P2 Lap is available
    if degree >= 2:
        R_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta_c, ufl.grad(u))
    else:
        R_u = ufl.dot(beta_c, ufl.grad(u))
    R_f = f_expr

    # Test function modification: beta . grad(v)
    v_supg = ufl.dot(beta_c, ufl.grad(v))

    a += tau * R_u * v_supg * ufl.dx
    L += tau * R_f * v_supg * ufl.dx

    # BC: u = sin(pi x) sin(pi y) = 0 on all boundaries
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda xx: np.ones(xx.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "ilu",
                       "ksp_rtol": 1e-10, "ksp_atol": 1e-12,
                       "ksp_max_it": 2000},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    ksp = problem.solver
    its = ksp.getIterationNumber()

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Verify error against manufactured solution
    u_ex_grid = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((u_grid - u_ex_grid) ** 2))
    print(f"RMS error vs exact on output grid: {err:.3e}", flush=True)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    import time
    spec = {
        "output": {
            "grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    t0 = time.time()
    out = solve(spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f} s")
    print(f"Shape: {out['u'].shape}")
