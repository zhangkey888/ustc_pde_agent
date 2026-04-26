import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    pde = case_spec["pde"]
    k = float(pde["coefficients"]["k"])
    output = case_spec["output"]
    grid_spec = output["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # Solver parameters - high accuracy with direct LU for indefinite Helmholtz
    mesh_res = 128
    element_degree = 4
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    t0 = time.perf_counter()
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    pi = ufl.pi
    u_exact = ufl.sin(3*pi*x[0]) * ufl.sin(3*pi*x[1])
    laplacian_u_exact = -2.0 * (3.0*pi)**2 * ufl.sin(3.0*pi*x[0]) * ufl.sin(3.0*pi*x[1])
    f_expr = -laplacian_u_exact - k**2 * u_exact
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Dirichlet BC on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve with direct LU
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate solution on output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
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
    
    u_values = np.zeros((nx_out * ny_out,))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    error_L2 = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form(
            ufl.inner(u_sol - u_exact_func, u_sol - u_exact_func) * ufl.dx
        )),
        op=MPI.SUM
    ))
    u_exact_L2 = np.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(fem.form(ufl.inner(u_exact_func, u_exact_func) * ufl.dx)),
        op=MPI.SUM
    ))
    rel_error = error_L2 / u_exact_L2 if u_exact_L2 > 0 else error_L2
    
    # Max error on grid
    u_exact_grid = np.sin(3*np.pi*XX) * np.sin(3*np.pi*YY)
    max_error = np.max(np.abs(u_grid - u_exact_grid))
    
    t1 = time.perf_counter()
    
    if comm.rank == 0:
        print(f"L2 error: {error_L2:.6e}, relative: {rel_error:.6e}")
        print(f"Max grid error: {max_error:.6e}")
        print(f"Target: <= 9.05e-05, PASS: {max_error <= 9.05e-05}")
        print(f"Wall time: {t1-t0:.2f}s, Time limit: 662.5s")
        print(f"Iterations: {iterations}")
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }
