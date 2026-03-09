import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    """Solve the Poisson equation with high-frequency source."""
    
    # Parse case_spec for any overrides
    pde_spec = case_spec.get("pde", {})
    coefficients = pde_spec.get("coefficients", {})
    kappa_val = coefficients.get("kappa", 1.0)
    
    # Output grid
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    mesh_resolution = 64
    element_degree = 2
    ksp_type_str = "cg"
    pc_type_str = "hypre"
    rtol_val = 1e-10
    
    comm = MPI.COMM_WORLD
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, 
                                      cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = sin(12*pi*x)*sin(10*pi*y)
    f_expr = ufl.sin(12 * ufl.pi * x[0]) * ufl.sin(10 * ufl.pi * x[1])
    
    # Coefficient kappa
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    
    # Bilinear and linear forms
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # Boundary conditions: u = 0 on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    # Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "ksp_rtol": str(rtol_val),
            "ksp_max_it": "1000",
        },
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    
    # Get iteration count
    iterations = problem.solver.getIterationNumber()
    
    # Evaluate solution on output grid
    u_grid = evaluate_on_grid(domain, u_sol, nx_out, ny_out)
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type_str,
        "pc_type": pc_type_str,
        "rtol": rtol_val,
        "iterations": iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }


def evaluate_on_grid(domain, u_func, nx, ny):
    """Evaluate a dolfinx Function on a uniform nx x ny grid over [0,1]x[0,1]."""
    
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((nx * ny, 3))
    points[:, 0] = XX.flatten()
    points[:, 1] = YY.flatten()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    u_values = np.full(nx * ny, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(nx * ny):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx, ny))
    
    return u_grid


if __name__ == "__main__":
    import time
    
    case_spec = {
        "pde": {
            "type": "poisson",
            "source": "sin(12*pi*x)*sin(10*pi*y)",
            "coefficients": {"kappa": 1.0},
        },
        "domain": {"type": "unit_square"},
        "output": {"nx": 50, "ny": 50},
    }
    
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    
    u_grid = result["u"]
    info = result["solver_info"]
    
    print(f"Solve time: {elapsed:.3f}s")
    print(f"Solution shape: {u_grid.shape}")
    print(f"Solution range: [{np.nanmin(u_grid):.6e}, {np.nanmax(u_grid):.6e}]")
    print(f"Any NaN: {np.any(np.isnan(u_grid))}")
    print(f"Solver info: {info}")
    
    # Check against analytical solution
    xs = np.linspace(0, 1, 50)
    ys = np.linspace(0, 1, 50)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    denom = 1.0 * ((12*np.pi)**2 + (10*np.pi)**2)
    u_exact = np.sin(12*np.pi*XX) * np.sin(10*np.pi*YY) / denom
    
    diff = u_grid - u_exact
    valid = ~np.isnan(diff)
    l2_err = np.sqrt(np.mean(diff[valid]**2))
    print(f"L2 grid error: {l2_err:.6e}")
