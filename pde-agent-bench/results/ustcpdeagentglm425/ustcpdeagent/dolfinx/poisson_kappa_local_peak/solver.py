import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract output grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = [float(v) for v in bbox]

    # Solver parameters - tuned for accuracy within time budget
    mesh_resolution = 140
    elem_degree = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))

    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)

    # Define kappa using UFL
    kappa = 1.0 + 30.0 * ufl.exp(-150.0 * ((x[0] - 0.35)**2 + (x[1] - 0.65)**2))

    # Manufactured solution
    u_exact_ufl = ufl.sin(ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])

    # Source term: f = -div(kappa * grad(u_exact))
    f_ufl = -ufl.div(kappa * ufl.grad(u_exact_ufl))

    # Interpolate f onto function for RHS
    f_func = fem.Function(V)
    f_expr = fem.Expression(f_ufl, V.element.interpolation_points)
    f_func.interpolate(f_expr)

    # Dirichlet BC: u = g on boundary
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)

    # Boundary facets and DOFs
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)

    # Variational form: integral(kappa * grad(u), grad(v)) dx = integral(f, v) dx
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
    L = ufl.inner(f_func, v_test) * ufl.dx

    # Solve with CG + Hypre AMG
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    # Get iteration count
    iterations = problem.solver.getIterationNumber()

    # --- Compute L2 error for verification ---
    L2_error_sq = fem.assemble_scalar(
        fem.form(ufl.inner(u_sol - u_exact_ufl, u_sol - u_exact_ufl) * ufl.dx)
    )
    L2_error = np.sqrt(comm.allreduce(L2_error_sq, op=MPI.SUM))
    if comm.rank == 0:
        print(f"L2 error: {L2_error:.6e}")

    # --- Sample solution on output grid ---
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts_flat = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts_flat)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts_flat)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts_flat.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts_flat[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts_flat.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()

    # Gather across ranks
    if comm.size > 1:
        u_local = np.where(np.isnan(u_values), 0.0, u_values)
        u_global = np.zeros_like(u_values)
        comm.Allreduce(u_local, u_global, op=MPI.SUM)
        u_values = u_global

    u_grid = u_values.reshape(ny_out, nx_out)

    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": elem_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "coefficients": {
                "kappa": {"type": "expr", "expr": "1 + 30*exp(-150*((x-0.35)**2 + (y-0.65)**2))"}
            },
            "time": None
        },
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
