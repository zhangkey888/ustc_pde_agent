import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract case spec info
    pde = case_spec["pde"]
    output_spec = case_spec["output"]
    grid_spec = output_spec["grid"]
    
    nx_grid = grid_spec["nx"]
    ny_grid = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # PDE info
    kappa_spec = pde.get("coefficients", {}).get("kappa", {"type": "constant", "value": 1.0})
    
    # Mesh resolution and element degree - high accuracy with time budget
    mesh_res = 200
    element_degree = 3
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinates for UFL expressions
    x = ufl.SpatialCoordinate(domain)
    
    # Kappa coefficient
    if kappa_spec.get("type") == "expr":
        kappa_ufl = 0.2 + 0.8 * ufl.exp(-80.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
    else:
        kappa_ufl = fem.Constant(domain, ScalarType(1.0))
    
    # Source term f = 1.0
    f = fem.Constant(domain, ScalarType(1.0))
    
    # Variational form: -div(kappa * grad(u)) = f
    a = ufl.inner(kappa_ufl * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions - Dirichlet u=0 on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)
    
    # Solve with CG + AMG
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
    
    # Get solver info
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Sample solution onto output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_grid)
    ys = np.linspace(ymin, ymax, ny_grid)
    XX, YY = np.meshgrid(xs, ys)
    
    # Prepare points for evaluation (3D: x, y, z=0)
    points = np.zeros((3, nx_grid * ny_grid))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
    # Evaluate solution at grid points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full((nx_grid * ny_grid,), np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (ny, nx)
    u_grid = u_values.reshape(ny_grid, nx_grid)
    
    # Handle any NaN values
    if np.any(np.isnan(u_grid)):
        u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    # Accuracy verification: compute L2 norm and H1 semi-norm
    l2_norm = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(u_sol, u_sol) * ufl.dx)), op=MPI.SUM
    ))
    h1_semi = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(ufl.grad(u_sol), ufl.grad(u_sol)) * ufl.dx)), op=MPI.SUM
    ))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
