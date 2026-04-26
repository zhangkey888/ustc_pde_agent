import math
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl

ScalarType = PETSc.ScalarType


def _sample_function_on_grid(domain, uh, grid_spec):
    nx = int(grid_spec["nx"])
    ny = int(grid_spec["ny"])
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = map(float, bbox)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx * ny, dtype=np.float64)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    values_local = np.full(nx * ny, np.nan, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        values_local[np.array(eval_map, dtype=np.int32)] = np.asarray(vals).reshape(-1)

    comm = domain.comm
    gathered = comm.gather(values_local, root=0)
    if comm.rank == 0:
        merged = np.full(nx * ny, np.nan, dtype=np.float64)
        for arr in gathered:
            mask = ~np.isnan(arr)
            merged[mask] = arr[mask]
        if np.isnan(merged).any():
            raise RuntimeError("Failed to evaluate solution at some requested output grid points.")
        return merged.reshape(ny, nx)
    return None


def _manufactured_rhs_ufl(x, eps, beta):
    u_exact = ufl.sin(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1])
    lap_u = -(4.0 * ufl.pi) ** 2 * u_exact - (3.0 * ufl.pi) ** 2 * u_exact
    grad_u = ufl.as_vector(
        [
            4.0 * ufl.pi * ufl.cos(4.0 * ufl.pi * x[0]) * ufl.sin(3.0 * ufl.pi * x[1]),
            3.0 * ufl.pi * ufl.sin(4.0 * ufl.pi * x[0]) * ufl.cos(3.0 * ufl.pi * x[1]),
        ]
    )
    f = -eps * lap_u + ufl.dot(beta, grad_u)
    return u_exact, f


def solve(case_spec: dict) -> dict:
    # ```DIAGNOSIS
    # equation_type: convection_diffusion
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: scalar
    # coupling: none
    # linearity: linear
    # time_dependence: steady
    # stiffness: stiff
    # dominant_physics: mixed
    # peclet_or_reynolds: high
    # solution_regularity: smooth
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution
    # ```
    #
    # ```METHOD
    # spatial_method: fem
    # element_or_basis: Lagrange_P2
    # stabilization: supg
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: gmres
    # preconditioner: ilu
    # special_treatment: none
    # pde_skill: convection_diffusion / reaction_diffusion / biharmonic
    # ```

    comm = MPI.COMM_WORLD

    eps_value = 0.05
    beta_vec = np.array([3.0, 3.0], dtype=np.float64)

    # Accuracy/time trade-off chosen for high-frequency manufactured solution with Pe~84.9
    n = 96
    degree = 2
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1.0e-9

    domain = mesh.create_unit_square(comm, n, n, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    eps = fem.Constant(domain, ScalarType(eps_value))
    beta = fem.Constant(domain, ScalarType(beta_vec))

    u_exact_ufl, f_ufl = _manufactured_rhs_ufl(x, eps, beta)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta) + 1.0e-16)
    tau = h / (2.0 * beta_norm)

    a_std = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx
    L_std = f_ufl * v * ufl.dx

    residual_u = -eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
    residual_v = ufl.dot(beta, ufl.grad(v))
    a_supg = tau * residual_u * residual_v * ufl.dx
    L_supg = tau * f_ufl * residual_v * ufl.dx

    a = a_std + a_supg
    L = L_std + L_supg

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(
        lambda X: np.sin(4.0 * np.pi * X[0]) * np.sin(3.0 * np.pi * X[1])
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options_prefix="convdiff_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1.0e-12,
            "ksp_max_it": 5000,
        },
    )

    uh = problem.solve()
    uh.x.scatter_forward()

    # Accuracy verification against analytical manufactured solution
    u_exact = fem.Function(V)
    u_exact.interpolate(
        lambda X: np.sin(4.0 * np.pi * X[0]) * np.sin(3.0 * np.pi * X[1])
    )

    err_fun = fem.Function(V)
    err_fun.x.array[:] = uh.x.array - u_exact.x.array
    err_fun.x.scatter_forward()

    l2_err_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_ref_local = fem.assemble_scalar(fem.form(ufl.inner(u_exact, u_exact) * ufl.dx))
    l2_err = math.sqrt(comm.allreduce(l2_err_local, op=MPI.SUM))
    l2_ref = math.sqrt(comm.allreduce(l2_ref_local, op=MPI.SUM))
    rel_l2_err = l2_err / max(l2_ref, 1.0e-16)

    ksp = problem.solver
    iterations = int(ksp.getIterationNumber())

    grid = case_spec["output"]["grid"]
    u_grid = _sample_function_on_grid(domain, uh, grid)

    solver_info = {
        "mesh_resolution": n,
        "element_degree": degree,
        "ksp_type": ksp.getType(),
        "pc_type": ksp.getPC().getType(),
        "rtol": float(rtol),
        "iterations": iterations,
        "l2_error": float(l2_err),
        "relative_l2_error": float(rel_l2_err),
        "stabilization": "SUPG",
    }

    if comm.rank == 0:
        return {"u": u_grid, "solver_info": solver_info}
    return {"u": None, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": None},
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(result["u"].shape)
        print(result["solver_info"])
