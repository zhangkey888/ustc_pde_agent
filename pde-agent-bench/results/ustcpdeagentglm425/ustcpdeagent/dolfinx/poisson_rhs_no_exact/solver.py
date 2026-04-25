import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parse case_spec
    pde = case_spec["pde"]
    coeffs = pde["coefficients"]
    kappa_val = coeffs.get("kappa", 0.5)

    # Output grid spec
    out_spec = case_spec["output"]
    grid = out_spec["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

    # Mesh resolution and element degree
    mesh_res = 128
    elem_deg = 3

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Source term as UFL expression
    x = ufl.SpatialCoordinate(domain)
    f_expr = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

    # Variational forms: -div(kappa * grad(u)) = f
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx

    # Boundary conditions: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

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
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        pts_array = np.array(points_on_proc)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_array, cells_array)
        u_values[eval_map] = vals.flatten()

    # Handle parallel: gather on all ranks
    u_values_local = np.nan_to_num(u_values, nan=0.0)
    comm.Barrier()
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values_local, u_values_global, op=MPI.SUM)
    u_grid = u_values_global.reshape(ny_out, nx_out)

    # Verify accuracy using analytical solution
    exact_coeff = 1.0 / (kappa_val * 13.0 * np.pi**2)
    u_exact_grid = exact_coeff * np.sin(3*np.pi*XX) * np.sin(2*np.pi*YY)
    error_grid = np.abs(u_grid - u_exact_grid)
    max_error = np.max(error_grid)
    l2_error = np.sqrt(np.mean(error_grid**2))

    if comm.rank == 0:
        print(f"Max error: {max_error:.6e}, L2 error: {l2_error:.6e}")
        print(f"Iterations: {iterations}")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }

    return {"u": u_grid, "solver_info": solver_info}
