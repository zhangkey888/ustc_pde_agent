import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract output grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox

    # Mesh resolution and element degree
    mesh_res = 256
    elem_deg = 2

    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))

    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)

    # Define kappa with two peaks
    kappa_expr = 1.0 + 15.0*ufl.exp(-200.0*((x[0]-0.25)**2 + (x[1]-0.25)**2)) + 15.0*ufl.exp(-200.0*((x[0]-0.75)**2 + (x[1]-0.75)**2))

    # Manufactured solution
    u_exact = ufl.exp(0.5*x[0])*ufl.sin(2*ufl.pi*x[1])

    # Compute source term: f = -div(kappa * grad(u_exact))
    f_expr = -ufl.div(kappa_expr * ufl.grad(u_exact))

    # Boundary condition: u = g on boundary
    g_expr = u_exact

    # Interpolate kappa into a Function for efficiency
    kappa_func = fem.Function(V)
    kappa_func.interpolate(fem.Expression(kappa_expr, V.element.interpolation_points))

    # Interpolate f into a Function
    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))

    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ufl.inner(kappa_func * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_func * v * ufl.dx

    # Boundary conditions - Dirichlet on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    # Interpolate boundary condition
    g_func = fem.Function(V)
    g_func.interpolate(fem.Expression(g_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(g_func, boundary_dofs)

    # Solver parameters
    ksp_type = "cg"
    pc_type = "hypre"
    rtol_val = 1e-10

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": rtol_val,
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()

    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    # Sample solution on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)

    points_on_proc = []
    cells_on_proc = []
    eval_indices = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_indices.append(i)

    u_values = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_indices] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Compute L2 error against exact solution for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    error_L2 = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx)),
        op=MPI.SUM
    ))
    u_norm = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form(u_exact_func**2 * ufl.dx)),
        op=MPI.SUM
    ))
    rel_error = error_L2 / u_norm if u_norm > 0 else error_L2
    
    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}, Relative L2 error: {rel_error:.6e}")

    result = {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_deg,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol_val,
            "iterations": iterations,
        }
    }

    return result
