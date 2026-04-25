import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Output grid specification
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    xmin, xmax, ymin, ymax = bbox

    # Mesh resolution and element degree - optimized for accuracy within time budget
    mesh_res = 290
    elem_deg = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    # Define kappa as a UFL expression
    x = ufl.SpatialCoordinate(domain)
    r2 = (x[0] - 0.55)**2 + (x[1] - 0.45)**2
    kappa_expr = 0.2 + ufl.exp(-120 * r2)
    
    # Analytical derivatives of kappa
    dkappa_dx = -240 * (x[0] - 0.55) * ufl.exp(-120 * r2)
    dkappa_dy = -240 * (x[1] - 0.45) * ufl.exp(-120 * r2)
    
    # Interpolate kappa into a function
    kappa_func = fem.Function(V)
    kappa_func.interpolate(fem.Expression(kappa_expr, V.element.interpolation_points))
    
    # Variational problem: -div(kappa * grad(u)) = f
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Source term derived from manufactured solution u = sin(pi*x)*sin(pi*y)
    pi = ufl.pi
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # f = 2*kappa*pi^2*sin(pi*x)*sin(pi*y) - pi*(dkappa_dx*cos(pi*x)*sin(pi*y) + dkappa_dy*sin(pi*x)*cos(pi*y))
    f_expr = (2 * kappa_expr * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
              - pi * (dkappa_dx * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
                      + dkappa_dy * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])))
    
    # Interpolate f into a function
    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_expr, V.element.interpolation_points))
    
    # Bilinear and linear forms
    a = ufl.inner(kappa_func * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_func, v) * ufl.dx
    
    # Dirichlet BC: u = 0 on boundary (since sin(pi*0)=sin(pi*1)=0)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # Solve with CG + Hypre AMG preconditioner
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
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
    pts = np.zeros((ny_out * nx_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    pts[:, 2] = 0.0
    
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
    
    u_grid = np.full((ny_out * nx_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
    
    # Handle parallel: gather on all procs
    if comm.size > 1:
        u_grid_local = u_grid.copy()
        u_grid_global = np.zeros_like(u_grid)
        from mpi4py import MPI as MPI4
        comm.Allreduce(u_grid_local, u_grid_global, op=MPI4.SUM)
        u_grid = u_grid_global
    
    u_grid = u_grid.reshape(ny_out, nx_out)
    
    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    error_L2 = comm.allreduce(
        fem.assemble_scalar(fem.form((u_sol - u_exact_func)**2 * ufl.dx)),
        op=MPI.SUM
    )
    error_L2 = np.sqrt(error_L2)
    
    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}, iterations: {iterations}")
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        },
        "pde": {
            "time": None
        }
    }
    import time
    t0 = time.perf_counter()
    result = solve(case_spec)
    t1 = time.perf_counter()
    print(f"Wall time: {t1-t0:.4f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"solver_info: {result['solver_info']}")
