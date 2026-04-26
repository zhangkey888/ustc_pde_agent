import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters from case_spec
    pde = case_spec["pde"]
    k_val = pde.get("k", 24.0)
    
    output_spec = case_spec["output"]
    grid_spec = output_spec["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Mesh resolution and element degree chosen to maximize accuracy within time budget
    # k=24, wavelength ~0.26. With P2 and h=1/64, kh/p ~ 0.13 (good resolution)
    mesh_resolution = 64
    element_degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define variational problem: -∇²u - k²u = f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    k = fem.Constant(domain, PETSc.ScalarType(k_val))
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = sin(5πx)sin(4πy)
    # -∇²u = (25π² + 16π²) sin(5πx)sin(4πy) = 41π² sin(5πx)sin(4πy)
    # f = -∇²u - k²u = (41π² - k²) sin(5πx)sin(4πy)
    f_coeff = (5.0 * np.pi)**2 + (4.0 * np.pi)**2 - k_val**2
    f_expr = f_coeff * ufl.sin(5.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    
    # Bilinear form: a(u,v) = ∫(∇u·∇v - k²uv)dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx
    
    # Linear form: L(v) = ∫f·v dx
    L = f_expr * v * ufl.dx
    
    # Dirichlet BC: u = sin(5πx)sin(4πy) on ∂Ω
    g_expr = ufl.sin(5.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(g_expr, V.element.interpolation_points))
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve with direct LU (MUMPS)
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="helm_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros(points.shape[1])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Compute L2 error for verification
    u_exact_expr = ufl.sin(5.0 * ufl.pi * x[0]) * ufl.sin(4.0 * ufl.pi * x[1])
    error_L2_sq = fem.assemble_scalar(
        fem.form(ufl.inner(u_sol - u_exact_expr, u_sol - u_exact_expr) * ufl.dx))
    error_L2 = np.sqrt(comm.allreduce(error_L2_sq, op=MPI.SUM))
    
    # Compute H1 semi-error
    error_H1_sq = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.grad(u_sol - u_exact_expr), 
                           ufl.grad(u_sol - u_exact_expr)) * ufl.dx))
    error_H1_sq = comm.allreduce(error_H1_sq, op=MPI.SUM)
    
    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}")
        print(f"H1 semi-error: {np.sqrt(error_H1_sq):.6e}")
        print(f"Solver iterations: {iterations}")
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": int(iterations),
    }
    
    # Check for time-dependent PDE fields
    if "time" in pde and pde["time"] is not None:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
