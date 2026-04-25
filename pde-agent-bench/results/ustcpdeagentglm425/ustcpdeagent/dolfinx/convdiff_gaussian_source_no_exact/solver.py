import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # --- Extract parameters ---
    pde = case_spec["pde"]
    eps = pde.get("diffusion", 0.02)
    beta_vec = pde.get("velocity", [8.0, 3.0])
    
    output_spec = case_spec["output"]
    grid_spec = output_spec["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # --- Mesh and function space ---
    mesh_res = 512
    elem_degree = 1
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, mesh_res, mesh_res, 
        cell_type=mesh.CellType.triangle
    )
    V = fem.functionspace(domain, ("Lagrange", elem_degree))
    
    # --- Define variational problem with SUPG ---
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: Gaussian
    f = ufl.exp(-250.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
    
    # Convection velocity
    beta = ufl.as_vector(beta_vec)
    
    # SUPG stabilization parameter
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.inner(beta, beta))
    Pe_local = beta_norm * h / (2.0 * eps)
    tau = h / (2.0 * beta_norm) * ufl.conditional(
        Pe_local > 1.0, 1.0 - 1.0/Pe_local, 0.0
    )
    
    # Standard Galerkin: eps*grad(u).grad(v) + (beta.grad(u))*v = f*v
    a_galerkin = (eps * ufl.inner(ufl.grad(u), ufl.grad(v)) 
                  + ufl.inner(ufl.dot(beta, ufl.grad(u)), v)) * ufl.dx
    L_galerkin = ufl.inner(f, v) * ufl.dx
    
    # SUPG stabilization
    a_supg = ufl.inner(
        ufl.dot(beta, ufl.grad(u)), 
        tau * ufl.dot(beta, ufl.grad(v))
    ) * ufl.dx
    L_supg = ufl.inner(f, tau * ufl.dot(beta, ufl.grad(v))) * ufl.dx
    
    a = a_galerkin + a_supg
    L = L_galerkin + L_supg
    
    # --- Boundary conditions (u = 0 on all boundaries) ---
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # --- Solve with LU direct solver ---
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
    
    iterations = 1  # Direct solver
    
    # --- Sample solution on output grid ---
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    # Build points array (3, N) for dolfinx
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
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
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    # Gather values across all processes
    u_values_global = np.zeros_like(u_values)
    domain.comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
    u_values_global = np.nan_to_num(u_values_global, nan=0.0)
    
    u_grid = u_values_global.reshape(ny_out, nx_out)
    
    # --- Solver info ---
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
    import time
    t0 = time.time()
    case_spec = {
        "pde": {
            "diffusion": 0.02,
            "velocity": [8.0, 3.0],
        },
        "output": {
            "grid": {
                "nx": 65,
                "ny": 65,
                "bbox": [0.0, 1.0, 0.0, 1.0]
            }
        }
    }
    result = solve(case_spec)
    u_grid = result["u"]
    t1 = time.time()
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution min: {u_grid.min():.8e}, max: {u_grid.max():.8e}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Total time: {t1-t0:.2f}s")
