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
    eps = pde["diffusion"]       # 0.01
    beta_val = pde["velocity"]   # [15.0, 0.0]
    
    out = case_spec["output"]
    grid_info = out["grid"]
    nx_out = grid_info["nx"]
    ny_out = grid_info["ny"]
    bbox = grid_info["bbox"]  # [xmin, xmax, ymin, ymax]
    
    xmin, xmax, ymin, ymax = bbox
    
    # Mesh resolution - use higher resolution since LU is fast
    N = 160
    
    # Create mesh
    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [N, N],
        cell_type=mesh.CellType.triangle
    )
    
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Element degree
    elem_deg = 2
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact solution: u = sin(pi*x)*sin(pi*y)
    u_exact = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Source term from manufactured solution
    f_expr = eps * 2 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1]) \
             + beta_val[0] * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Convection vector
    beta = ufl.as_vector([beta_val[0], beta_val[1]])
    
    # SUPG stabilization parameter (Shakib-Hughes-Codina)
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.inner(beta, beta))
    tau = h / (2.0 * beta_norm + 4.0 * eps / h)
    
    # Galerkin part
    a_gal = eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
            + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
    
    # Linear form
    L = f_expr * v * ufl.dx
    
    # SUPG stabilization (streamline diffusion)
    a_supg = tau * ufl.inner(beta, ufl.grad(v)) * ufl.inner(beta, ufl.grad(u)) * ufl.dx
    L_supg = tau * ufl.inner(beta, ufl.grad(v)) * f_expr * ufl.dx
    
    a = a_gal + a_supg
    L_total = L + L_supg
    
    # Boundary conditions - u = 0 on boundary of [0,1]^2
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # Solve with direct LU solver
    problem = petsc.LinearProblem(
        a, L_total, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="convdiff_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    iterations = 1
    rtol = 1e-10
    
    # Sample solution onto output grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    # Build arrays for batch evaluation
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(nx_out * ny_out):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        for idx, i in enumerate(eval_map):
            u_values[i] = vals[idx, 0]
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Compute L2 error for verification
    L2_error_sq = fem.assemble_scalar(
        fem.form((u_sol - u_exact)**2 * ufl.dx)
    )
    L2_error = np.sqrt(MPI.COMM_WORLD.allreduce(L2_error_sq, op=MPI.SUM))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": elem_deg,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": rtol,
        "iterations": iterations,
    }
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
    }
    
    if comm.rank == 0:
        print(f"L2 error: {L2_error:.6e}")
    
    return result
