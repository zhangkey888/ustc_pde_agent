import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    pde = case_spec["pde"]
    epsilon = pde["coeffs"]["epsilon"]
    beta_vec = pde["coeffs"]["beta"]
    
    # Output grid specs
    out = case_spec["output"]
    nx_out = out["grid"]["nx"]
    ny_out = out["grid"]["ny"]
    bbox = out["grid"]["bbox"]
    
    # Parameters - balanced for speed and accuracy
    mesh_res = 90
    elem_degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution
    u_exact = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    
    beta = ufl.as_vector(beta_vec)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    
    # Source term from manufactured solution
    grad_u_exact = ufl.grad(u_exact)
    laplacian_u_exact = ufl.div(grad_u_exact)
    f_source = -epsilon * laplacian_u_exact + ufl.dot(beta, grad_u_exact)
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    Pe_loc = beta_mag * h / (2.0 * epsilon)
    tau_supg = h / (2.0 * beta_mag) * ufl.conditional(Pe_loc > 1.0, 1.0 - 1.0/Pe_loc, 0.0)
    
    # Bilinear form with SUPG
    a = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) 
         + ufl.dot(beta, ufl.grad(u)) * v 
         + tau_supg * ufl.dot(beta, ufl.grad(u)) * ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    # Linear form with SUPG
    L = (f_source * v 
         + tau_supg * f_source * ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    # Dirichlet BC from exact solution
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc_func, boundary_dofs)
    
    # Solve
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-12,
        },
        petsc_options_prefix="convdiff_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    iterations = problem.solver.getIterationNumber()
    
    # L2 error verification
    L2_err_form = fem.form(ufl.inner(u_sol - u_exact, u_sol - u_exact) * ufl.dx)
    L2_err_local = fem.assemble_scalar(L2_err_form)
    L2_err = np.sqrt(domain.comm.allreduce(L2_err_local, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {L2_err:.6e}, iterations: {iterations}")
    
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
    
    u_values = np.zeros(nx_out * ny_out, dtype=np.float64)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {"coeffs": {"epsilon": 0.05, "beta": [3.0, 3.0]}, "time": None},
        "output": {"grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    result = solve(case_spec)
    print(f"Shape: {result['u'].shape}, Info: {result['solver_info']}")
