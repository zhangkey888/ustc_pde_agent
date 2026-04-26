import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Parse case spec
    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    
    nx_out = grid.get("nx", 50)
    ny_out = grid.get("ny", 50)
    bbox = grid.get("bbox", [0, 1, 0, 1])
    
    # Material parameters
    E = 1.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    # Mesh resolution - use finer mesh to resolve tanh layer
    mesh_res = 80
    
    # Create mesh
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Function space - P2 vector elements
    elem_degree = 2
    V = fem.functionspace(msh, ("Lagrange", elem_degree, (gdim,)))
    
    # Define exact solution symbolically using UFL
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_ex = ufl.as_vector([
        ufl.tanh(6*(x[1] - 0.5)) * ufl.sin(pi * x[0]),
        0.1 * ufl.sin(2*pi * x[0]) * ufl.sin(pi * x[1])
    ])
    
    # Strain and stress definitions
    def eps(v):
        return ufl.sym(ufl.grad(v))
    
    def sigma(v):
        return 2.0 * mu * eps(v) + lam * ufl.tr(eps(v)) * ufl.Identity(gdim)
    
    # Source term from manufactured solution: f = -div(sigma(u_ex))
    f_source = -ufl.div(sigma(u_ex))
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f_source, v) * ufl.dx
    
    # Dirichlet BCs from exact solution on all boundaries
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve with CG + AMG
    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "cg",
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg",
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
            "ksp_max_it": 500,
        },
        petsc_options_prefix="elasticity_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get solver info
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Compute L2 error for verification
    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    
    error_expr = ufl.inner(u_sol - u_ex_func, u_sol - u_ex_func) * ufl.dx
    error_form = fem.form(error_expr)
    error_local = fem.assemble_scalar(error_form)
    error_l2 = np.sqrt(msh.comm.allreduce(error_local, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {error_l2:.6e}")
        print(f"KSP iterations: {iterations}")
        print(f"Mesh resolution: {mesh_res}, Element degree: {elem_degree}")
    
    # Sample on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(msh, tdim)
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
    
    u_values = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals
    
    # Communicate results in parallel
    if comm.size > 1:
        # Gather all results
        all_u = np.zeros_like(u_values)
        comm.Allreduce(u_values, all_u, op=MPI.SUM)
        # Fix NaN: where all ranks have nan, result is 0, need to restore nan
        u_values = all_u
    
    # Compute displacement magnitude
    magnitude = np.linalg.norm(u_values, axis=1).reshape(ny_out, nx_out)
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": rtol,
        "iterations": int(iterations),
    }
    
    # Check for time dependence
    time_info = pde.get("time", None)
    if time_info:
        solver_info["dt"] = 0.0
        solver_info["n_steps"] = 0
        solver_info["time_scheme"] = "none"
    
    return {
        "u": magnitude,
        "solver_info": solver_info,
    }
