import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Extract case parameters
    pde = case_spec["pde"]
    E = pde["parameters"]["E"]
    nu = pde["parameters"]["nu"]
    
    output_grid = case_spec["output"]["grid"]
    nx_out = output_grid["nx"]
    ny_out = output_grid["ny"]
    bbox = output_grid["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Material parameters
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    # Mesh resolution - use finer mesh for better accuracy
    mesh_res = 72
    elem_degree = 2  # P2 required for nu > 0.4
    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim
    
    # Vector function space - P2 to avoid volumetric locking
    V = fem.functionspace(domain, ("Lagrange", elem_degree, (gdim,)))
    
    # Exact solution (manufactured)
    x = ufl.SpatialCoordinate(domain)
    u_exact = ufl.as_vector([
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    
    # Strain and stress
    def eps(v):
        return ufl.sym(ufl.grad(v))
    
    def sigma(v):
        return 2.0 * mu * eps(v) + lam * ufl.tr(eps(v)) * ufl.Identity(gdim)
    
    # Body force from manufactured solution: f = -div(sigma(u_exact))
    f_body = -ufl.div(sigma(u_exact))
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f_body, v) * ufl.dx
    
    # Boundary conditions - all Dirichlet (entire boundary)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve with direct LU solver
    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="elasticity_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get solver info
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Compute L2 error for verification
    u_err_func = fem.Function(V)
    u_err_func.interpolate(
        fem.Expression(u_exact, V.element.interpolation_points)
    )
    error_form = fem.form(ufl.inner(u_sol - u_err_func, u_sol - u_err_func) * ufl.dx)
    l2_error_sq = domain.comm.allreduce(fem.assemble_scalar(error_form), op=MPI.SUM)
    l2_error = np.sqrt(l2_error_sq) if l2_error_sq > 0 else 0.0
    
    # Compute exact solution L2 norm for relative error
    exact_norm_form = fem.form(ufl.inner(u_err_func, u_err_func) * ufl.dx)
    exact_norm_sq = domain.comm.allreduce(fem.assemble_scalar(exact_norm_form), op=MPI.SUM)
    exact_l2 = np.sqrt(exact_norm_sq)
    rel_error = l2_error / exact_l2 if exact_l2 > 0 else l2_error
    
    if MPI.COMM_WORLD.rank == 0:
        print(f"L2 error: {l2_error:.6e}, Relative: {rel_error:.6e}")
    
    # Sample on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((nx_out * ny_out, gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    # Allreduce across processes
    u_values_flat = u_values.copy()
    is_nan = np.isnan(u_values_flat)
    u_values_flat[is_nan] = 0.0
    u_global = np.zeros_like(u_values_flat)
    domain.comm.Allreduce(u_values_flat, u_global, op=MPI.SUM)
    
    # Compute displacement magnitude
    magnitude = np.linalg.norm(u_global, axis=1).reshape(ny_out, nx_out)
    
    return {
        "u": magnitude,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": rtol,
            "iterations": iterations,
        }
    }
