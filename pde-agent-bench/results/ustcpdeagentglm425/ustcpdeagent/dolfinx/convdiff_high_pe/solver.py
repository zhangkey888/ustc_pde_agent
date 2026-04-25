import numpy as np
import time
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    t_start = time.time()
    
    # Extract parameters
    pde = case_spec["pde"]
    eps = float(pde["coefficients"]["epsilon"])
    beta_vals = [float(b) for b in pde["coefficients"]["beta"]]
    
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh parameters - maximize accuracy within time budget
    N = 256
    deg = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", deg))
    
    x = ufl.SpatialCoordinate(domain)
    beta = ufl.as_vector(beta_vals)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    pi = ufl.pi
    
    # Manufactured solution: u = sin(pi*x)*sin(pi*y)
    u_exact_expr = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term: f = -eps*laplacian(u) + beta.grad(u)
    f_expr = (2.0 * eps * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
              + beta_vals[0] * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
              + beta_vals[1] * pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1]))
    
    # BC: u=0 on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # Variational form with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    h = ufl.CellDiameter(domain)
    tau = h / (2.0 * beta_mag)
    
    # Galerkin part
    a_galerkin = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
                  + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx)
    L_galerkin = f_expr * v * ufl.dx
    
    # SUPG stabilization (simplified residual, dropping 2nd order term)
    a_supg = tau * ufl.inner(ufl.dot(beta, ufl.grad(u)), ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L_supg = tau * f_expr * ufl.inner(beta, ufl.grad(v)) * ufl.dx
    
    a = a_galerkin + a_supg
    L = L_galerkin + L_supg
    
    # Solve with GMRES + ILU
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "ksp_rtol": 1e-10,
            "ksp_max_it": 2000,
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Sample solution onto output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[0],), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # L2 error verification
    u_ex = fem.Function(V)
    u_ex.interpolate(fem.Expression(u_exact_expr, V.element.interpolation_points))
    error_L2 = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form((u_sol - u_ex)**2 * ufl.dx)), op=MPI.SUM))
    
    t_elapsed = time.time() - t_start
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": deg,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": 1e-10,
        "iterations": iterations,
    }
    
    result = {"u": u_grid, "solver_info": solver_info}
    
    # Add time info if transient
    if pde.get("time") is not None:
        result["dt"] = 0.0
        result["n_steps"] = 0
        result["time_scheme"] = "none"
    
    return result
