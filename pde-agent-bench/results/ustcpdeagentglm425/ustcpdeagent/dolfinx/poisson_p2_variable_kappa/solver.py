import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    comm = MPI.COMM_WORLD
    
    # Extract output grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Mesh resolution and element degree
    mesh_res = 240
    elem_degree = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Define function space
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # Define spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Variable coefficient kappa
    kappa = 1.0 + 0.4 * ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Manufactured solution
    u_exact_ufl = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])
    
    # Source term: f = -div(kappa * grad(u_exact))
    grad_u_exact = ufl.grad(u_exact_ufl)
    f = -ufl.div(kappa * grad_u_exact)
    
    # Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx
    
    # Dirichlet BC
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve using CG with HYPRE AMG preconditioner
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
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get solver iterations
    ksp = problem.solver
    iterations = ksp.getIterationNumber()
    
    # Compute L2 error for verification
    L2_error = fem.assemble_scalar(fem.form((u_sol - u_exact_ufl)**2 * ufl.dx))
    L2_error = np.sqrt(comm.allreduce(L2_error, op=MPI.SUM))
    
    t1 = time.time()
    
    if comm.rank == 0:
        print(f"L2 error: {L2_error:.6e}")
        print(f"Solver iterations: {iterations}")
        print(f"Solve time: {t1-t0:.3f}s")
    
    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    u_values = np.zeros(points.shape[0])
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    if comm.size > 1:
        recv_buf = np.zeros_like(u_values)
        comm.Allreduce(u_values, recv_buf, op=MPI.SUM)
        u_grid = recv_buf.reshape(ny_out, nx_out)
    else:
        u_grid = u_values.reshape(ny_out, nx_out)
    
    t2 = time.time()
    if comm.rank == 0:
        print(f"Total time: {t2-t0:.3f}s")
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_degree,
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
        }
    }
    result = solve(case_spec)
    print(f"Output shape: {result['u'].shape}")
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys)
    exact = np.sin(2*np.pi*XX) * np.sin(2*np.pi*YY)
    grid_error = np.max(np.abs(result['u'] - exact))
    print(f"Max grid error: {grid_error:.6e}")
