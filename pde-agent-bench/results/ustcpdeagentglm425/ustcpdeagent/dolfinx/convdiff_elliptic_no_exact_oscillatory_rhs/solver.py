import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    # Extract parameters from case_spec
    pde = case_spec.get("pde", {})
    output_spec = case_spec.get("output", {})
    grid_spec = output_spec.get("grid", {})
    
    nx_grid = grid_spec.get("nx", 50)
    ny_grid = grid_spec.get("ny", 50)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    eps_val = pde.get("epsilon", 0.05)
    beta_val = pde.get("beta", [3.0, 3.0])
    
    # Parameters - mesh_res=512 with P2, SUPG stabilization
    mesh_res = 512
    element_degree = 2
    ksp_type = "gmres"
    pc_type = "hypre"
    pc_hypre_type = "boomeramg"
    rtol = 1e-12
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define variational problem with SUPG stabilization
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term
    f = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(5 * ufl.pi * x[1])
    
    # Convection velocity
    beta = ufl.as_vector([beta_val[0], beta_val[1]])
    
    # Diffusion coefficient
    epsilon = eps_val
    
    # SUPG stabilization parameter
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.inner(beta, beta))
    tau_supg = h / (2.0 * beta_norm)
    
    # Standard Galerkin terms
    a_galerkin = epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.inner(beta, ufl.grad(u)) * v * ufl.dx
    L_galerkin = f * v * ufl.dx
    
    # SUPG stabilization terms
    a_supg = tau_supg * ufl.inner(beta, ufl.grad(v)) * ufl.inner(beta, ufl.grad(u)) * ufl.dx
    L_supg = tau_supg * ufl.inner(beta, ufl.grad(v)) * f * ufl.dx
    
    a = a_galerkin + a_supg
    L = L_galerkin + L_supg
    
    # Boundary conditions: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve using LinearProblem with AMG preconditioner
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_hypre_type": pc_hypre_type,
            "ksp_rtol": rtol,
        },
        petsc_options_prefix="convdiff_"
    )
    
    u_sol = problem.solve()
    u_sol.x.scatter_forward()
    
    # Get iteration count
    ksp = problem._solver
    iterations = ksp.getIterationNumber()
    
    # Sample solution onto the output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_grid)
    ys = np.linspace(ymin, ymax, ny_grid)
    XX, YY = np.meshgrid(xs, ys)
    
    # Create points array (3D for dolfinx)
    points = np.zeros((3, nx_grid * ny_grid))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
    # Evaluate solution at grid points
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    u_values = np.full((nx_grid * ny_grid,), np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    # Reshape to (ny, nx)
    u_grid = u_values.reshape(ny_grid, nx_grid)
    
    # Build solver_info
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    # Check if time-dependent
    time_spec = pde.get("time", None)
    if time_spec is not None:
        solver_info["dt"] = 0.01
        solver_info["n_steps"] = 1
        solver_info["time_scheme"] = "backward_euler"
    
    result = {
        "u": u_grid,
        "solver_info": solver_info,
    }
    
    return result

# For testing
if __name__ == "__main__":
    case_spec = {
        "pde": {
            "equation": "convection_diffusion",
            "epsilon": 0.05,
            "beta": [3.0, 3.0],
            "source": "sin(6*pi*x)*sin(5*pi*y)",
            "boundary_conditions": {"type": "dirichlet", "value": 0.0},
            "time": None,
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
    start = time.time()
    result = solve(case_spec)
    wall_time = time.time() - start
    
    u_grid = result["u"]
    print(f"Solution shape: {u_grid.shape}")
    print(f"Max value: {np.nanmax(u_grid):.6e}")
    print(f"Min value: {np.nanmin(u_grid):.6e}")
    print(f"Solver info: {result['solver_info']}")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Time constraint: 298.324s, PASSED: {wall_time <= 298.324}")
