import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Extract case parameters
    pde = case_spec["pde"]
    eps_val = pde.get("diffusion", 0.01)
    beta_val = pde.get("convection", [20.0, 0.0])
    
    # Output grid parameters
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # Mesh resolution and element degree
    mesh_res = 128
    elem_degree = 2
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # Parameters
    epsilon = PETSc.ScalarType(eps_val)
    beta = ufl.as_vector(beta_val)
    beta_norm_val = np.sqrt(beta_val[0]**2 + beta_val[1]**2)
    
    # Manufactured solution: u = sin(pi*x)*sin(pi*y)
    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    u_exact_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term derived from manufactured solution
    # f = -eps*laplacian(u) + beta.grad(u)
    # laplacian(u) = -2*pi^2*sin(pi*x)*sin(pi*y)
    # beta.grad(u) = beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    f_val = (2.0 * eps_val * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
             + beta_val[0] * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
             + beta_val[1] * pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1]))
    
    # Boundary conditions - all Dirichlet (u=0 on boundary for this solution)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Standard Galerkin terms
    a_galerkin = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx 
                  + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx)
    L_galerkin = ufl.inner(f_val, v) * ufl.dx
    
    # SUPG stabilization parameter (Shakib et al. formula)
    h = ufl.CellDiameter(domain)
    tau = 1.0 / ufl.sqrt((2.0 * beta_norm_val / h)**2 + 12.0 * eps_val / h**2)
    
    # SUPG terms (CORRECTED SIGNS)
    # Residual R(u) = -eps*Delta(u) + beta.grad(u) - f
    # SUPG adds: tau * (beta.grad(v), R(u)) dx to the weak form
    # Bilinear addition: tau * (beta.grad(v), -eps*Delta(u) + beta.grad(u)) dx
    # Linear addition: tau * (beta.grad(v), f) dx (NOT negative!)
    
    a_supg = tau * ufl.inner(ufl.dot(beta, ufl.grad(v)),
                             -epsilon * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))) * ufl.dx
    
    L_supg = tau * ufl.inner(ufl.dot(beta, ufl.grad(v)), f_val) * ufl.dx
    
    a = a_galerkin + a_supg
    L = L_galerkin + L_supg
    
    # Solve
    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "pc_factor_levels": 5,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
            "ksp_max_it": 500,
        },
        petsc_options_prefix="convdiff_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get solver iteration count
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Evaluate solution on output grid
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
    
    u_values = np.full((pts_flat.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    
    error_L2 = fem.assemble_scalar(
        fem.form((u_sol - u_exact_func)**2 * ufl.dx)
    )
    error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(error_L2, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}")
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
        "ksp_type": "gmres",
        "pc_type": "ilu",
        "rtol": rtol,
        "iterations": int(iterations),
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"diffusion": 0.01, "convection": [20.0, 0.0]},
        "output": {"grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
    print(f"Solver info: {result['solver_info']}")
