import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    eps = 0.05
    beta = np.array([3.0, 1.0])
    
    # Grid output spec
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh resolution and element degree
    mesh_res = 192
    elem_deg = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    # Exact solution: u = sin(2*pi*(x+y)) * sin(pi*(x-y))
    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(2*ufl.pi*(x[0]+x[1])) * ufl.sin(ufl.pi*(x[0]-x[1]))
    
    # Gradient of exact solution
    grad_u_exact = ufl.grad(u_exact_ufl)
    
    # Source term: f = -eps * div(grad(u)) + beta . grad(u)
    laplacian_u = ufl.div(grad_u_exact)
    beta_vec = ufl.as_vector(beta)
    convection = ufl.dot(beta_vec, grad_u_exact)
    f_ufl = -eps * laplacian_u + convection
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Cell diameter for SUPG
    h = ufl.CellDiameter(domain)
    
    # SUPG stabilization parameter tau
    beta_norm = ufl.sqrt(ufl.dot(beta_vec, beta_vec))
    Pe_loc = beta_norm * h / (2.0 * eps)
    tau_supg = h / (2.0 * beta_norm) * (1.0 / ufl.tanh(Pe_loc) - 1.0 / Pe_loc)
    
    # Standard Galerkin part
    a_gal = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    
    # SUPG stabilization: tau * (beta . grad(v)) * (beta . grad(u))
    a_supg = ufl.inner(tau_supg * ufl.dot(beta_vec, ufl.grad(v)), ufl.dot(beta_vec, ufl.grad(u))) * ufl.dx
    
    # SUPG diffusion: tau * (beta . grad(v)) * (-eps * laplacian(u))
    a_supg_diff = ufl.inner(tau_supg * ufl.dot(beta_vec, ufl.grad(v)), -eps * ufl.div(ufl.grad(u))) * ufl.dx
    
    a = a_gal + a_supg + a_supg_diff
    
    # RHS
    L_gal = ufl.inner(f_ufl, v) * ufl.dx
    L_supg = ufl.inner(tau_supg * ufl.dot(beta_vec, ufl.grad(v)), f_ufl) * ufl.dx
    L = L_gal + L_supg
    
    # Boundary conditions - Dirichlet on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl, V.element.interpolation_points)
    u_bc_func.interpolate(u_bc_expr)
    
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # Solve with direct LU
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="convdiff_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    iterations = 1
    
    # Sample solution on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
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
    
    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_values_local = u_values.copy()
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values_local, u_values_global, op=MPI.SUM)
    
    u_grid = u_values_global.reshape(ny_out, nx_out)
    
    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(u_bc_expr)
    error_L2 = fem.assemble_scalar(fem.form(ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx))
    error_L2 = np.sqrt(comm.allreduce(error_L2, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}")
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_deg,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }

if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        },
        "pde": {}
    }
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
