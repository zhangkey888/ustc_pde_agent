import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract PDE parameters
    pde = case_spec.get("pde", {})
    eps = pde.get("diffusion", 0.05)
    beta_vec = pde.get("convection", [4.0, 2.0])
    
    # Output grid parameters
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Mesh resolution and element degree
    mesh_res = 256
    elem_deg = 2
    
    # Create mesh (quadrilateral)
    domain = mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.quadrilateral
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    # PDE coefficients
    beta = ufl.as_vector(beta_vec)
    
    # Spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u = sin(2*pi*x)*sin(pi*y)
    u_exact_ufl = ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    
    # Source term: f = -eps * laplacian(u) + beta . grad(u)
    f_ufl = (eps * 5 * ufl.pi**2 * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
             + beta_vec[0] * 2 * ufl.pi * ufl.cos(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
             + beta_vec[1] * ufl.pi * ufl.sin(2*ufl.pi*x[0]) * ufl.cos(ufl.pi*x[1]))
    
    # Check mesh Peclet number for SUPG decision
    h_approx = 1.0 / mesh_res
    beta_magnitude = np.sqrt(beta_vec[0]**2 + beta_vec[1]**2)
    pe_h = beta_magnitude * h_approx / (2 * eps)
    use_supg = pe_h > 2.0  # for P2 elements, effective Pe is lower
    
    # Boundary condition: u = sin(2*pi*x)*sin(pi*y) on entire boundary
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Trial and test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Variational form (Galerkin)
    a = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx)
    L = f_ufl * v * ufl.dx
    
    # Add SUPG stabilization if needed
    if use_supg:
        h = ufl.CellDiameter(domain)
        beta_norm_ufl = ufl.sqrt(beta[0]**2 + beta[1]**2)
        tau = h / (2.0 * beta_norm_ufl)
        a += tau * ufl.inner(ufl.dot(beta, ufl.grad(u)), ufl.dot(beta, ufl.grad(v))) * ufl.dx
        L += tau * f_ufl * ufl.inner(beta, ufl.grad(v)) * ufl.dx
    
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
    
    iterations = 1  # direct solver
    
    # Sample solution on output grid
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
    
    u_values = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr).flatten()
        for idx, gi in enumerate(eval_map):
            u_values[gi] = vals[idx]
    
    # Gather across processes if parallel
    u_values_global = np.zeros_like(u_values)
    comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
    
    u_grid = u_values_global.reshape(ny_out, nx_out)
    
    # Compute L2 error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    error_sq = fem.assemble_scalar(
        fem.form((u_sol - u_exact_func)**2 * ufl.dx)
    )
    l2_error = np.sqrt(max(comm.allreduce(error_sq, op=MPI.SUM), 0.0))
    
    H1_semi_sq = fem.assemble_scalar(
        fem.form(ufl.inner(ufl.grad(u_sol - u_exact_func), ufl.grad(u_sol - u_exact_func)) * ufl.dx)
    )
    h1_error = np.sqrt(max(comm.allreduce(H1_semi_sq, op=MPI.SUM), 0.0))
    
    if comm.rank == 0:
        print(f"Pe_h={pe_h:.4f}, SUPG={'ON' if use_supg else 'OFF'}")
        print(f"L2 error: {l2_error:.6e}, H1 semi-error: {h1_error:.6e}")
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": elem_deg,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": int(iterations),
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "diffusion": 0.05,
            "convection": [4.0, 2.0],
        },
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"solver_info: {result['solver_info']}")
    print(f"u min/max: {result['u'].min():.6e} / {result['u'].max():.6e}")
