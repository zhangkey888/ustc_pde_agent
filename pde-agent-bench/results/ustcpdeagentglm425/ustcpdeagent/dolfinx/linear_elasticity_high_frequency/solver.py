import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Extract case parameters
    pde = case_spec["pde"]
    params = pde["parameters"]
    E_val = float(params["E"])
    nu_val = float(params["nu"])
    
    # Material parameters
    mu_val = E_val / (2.0 * (1.0 + nu_val))
    lam_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))
    
    # Output grid specs
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Choose mesh resolution and element degree
    mesh_res = 240
    elem_deg = 2
    
    # Create mesh
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Vector function space
    V = fem.functionspace(msh, ("Lagrange", elem_deg, (gdim,)))
    
    # Define exact solution as UFL expression
    x = ufl.SpatialCoordinate(msh)
    u1_exact = ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    u2_exact = ufl.cos(3 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    u_exact = ufl.as_vector([u1_exact, u2_exact])
    
    # Strain and stress
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u):
        return 2.0 * mu_val * eps(u) + lam_val * ufl.tr(eps(u)) * ufl.Identity(gdim)
    
    # Source term from manufactured solution: f = -div(sigma(u_exact))
    f_expr = -ufl.div(sigma(u_exact))
    
    # Variational form
    u_trial = ufl.TrialFunction(V)
    v_test = ufl.TestFunction(V)
    
    a = ufl.inner(sigma(u_trial), eps(v_test)) * ufl.dx
    
    # Interpolate f onto a Function for the RHS
    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    L = ufl.inner(f_func, v_test) * ufl.dx
    
    # Dirichlet BCs on entire boundary from exact solution
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
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
            "ksp_max_it": 500,
        },
        petsc_options_prefix="elasticity_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Sample displacement magnitude on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    # Point evaluation
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    # Displacement magnitude
    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)
    
    # Gather across processes if parallel
    if comm.size > 1:
        magnitude_local = magnitude.copy()
        magnitude = np.zeros_like(magnitude_local)
        comm.Allreduce(magnitude_local, magnitude, op=MPI.SUM)
    
    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    error_expr = ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
    error_form = fem.form(error_expr)
    l2_error_sq = fem.assemble_scalar(error_form)
    l2_error = np.sqrt(comm.allreduce(l2_error_sq, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {l2_error:.6e}")
        print(f"KSP iterations: {iterations}")
    
    result = {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_deg,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }
    
    # Add time info if present
    if "time" in pde:
        result["solver_info"]["dt"] = 0.0
        result["solver_info"]["n_steps"] = 0
        result["solver_info"]["time_scheme"] = "none"
    
    return result
