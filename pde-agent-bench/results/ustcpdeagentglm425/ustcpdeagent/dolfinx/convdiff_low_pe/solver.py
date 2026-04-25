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
    eps = pde.get("epsilon", 0.2)
    beta = pde.get("beta", [1.0, 0.5])
    output = case_spec["output"]
    grid = output["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Mesh resolution and element degree
    mesh_res = 256
    elem_deg = 2
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution: u = sin(pi*x)*sin(pi*y)
    # Source term: f = -eps*laplacian(u) + beta.grad(u)
    # f = 2*eps*pi^2*sin(pi*x)*sin(pi*y) + beta[0]*pi*cos(pi*x)*sin(pi*y) + beta[1]*pi*sin(pi*x)*cos(pi*y)
    
    pi = np.pi
    f_expr = (2.0 * eps * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
              + beta[0] * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
              + beta[1] * pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1]))
    
    # Convection vector
    beta_vec = ufl.as_vector([beta[0], beta[1]])
    
    # Bilinear form: eps*grad(u).grad(v) + (beta.grad(u))*v
    a = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(ufl.dot(beta_vec, ufl.grad(u)), v) * ufl.dx
    
    # Linear form
    L = f_expr * v * ufl.dx
    
    # Boundary conditions: u = 0 on all boundaries (sin(0)=sin(pi)=0)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # Solve with direct LU solver for speed
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
    
    # Compute L2 error for verification
    u_exact_expr = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    L2_error = fem.assemble_scalar(fem.form((u_sol - u_exact_expr)**2 * ufl.dx))
    L2_error = np.sqrt(comm.allreduce(L2_error, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 error: {L2_error:.6e}")
    
    # Sample solution on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    # Points shape (3, N)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    # Find cells and evaluate
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    # Gather from all processes
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
    
    u_grid = u_values_global.reshape(ny_out, nx_out)
    
    # Build solver_info
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
        "pde": {"epsilon": 0.2, "beta": [1.0, 0.5]},
        "output": {
            "grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}
        }
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"solver_info: {result['solver_info']}")
    
    nx_out = 64
    ny_out = 64
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    err = np.sqrt(np.mean((result['u'] - u_exact)**2))
    print(f"RMS error on output grid: {err:.6e}")
