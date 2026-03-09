import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict = None) -> dict:
    k_val = 5.0
    nx_out = 50
    ny_out = 50
    
    if case_spec is not None:
        pde = case_spec.get("pde", {})
        k_val = pde.get("wavenumber", k_val)
        output = case_spec.get("output", {})
        nx_out = output.get("nx", nx_out)
        ny_out = output.get("ny", ny_out)
    
    comm = MPI.COMM_WORLD
    N = 48
    degree = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    pi_val = np.pi
    k2 = PETSc.ScalarType(k_val**2)
    
    f_expr = (2.0 * pi_val**2 - k_val**2) * ufl.sin(pi_val * x[0]) * ufl.sin(pi_val * x[1])
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.zeros_like(x[0]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    ksp_type = "preonly"
    pc_type = "lu"
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    iterations = problem.solver.getIterationNumber()
    
    xg = np.linspace(0, 1, nx_out)
    yg = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xg, yg, indexing='ij')
    points_2d = np.vstack([XX.ravel(), YY.ravel()])
    points_3d = np.vstack([points_2d, np.zeros(points_2d.shape[1])])
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d.T)
    
    u_grid = np.full(points_3d.shape[1], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points_3d.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
    
    u_grid = u_grid.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 1e-8,
            "iterations": iterations,
        }
    }

if __name__ == "__main__":
    import time
    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0
    u_grid = result["u"]
    nx, ny = u_grid.shape
    xg = np.linspace(0, 1, nx)
    yg = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xg, yg, indexing='ij')
    u_exact = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    max_error = np.max(np.abs(u_grid - u_exact))
    print(f"Time: {elapsed:.3f}s")
    print(f"L2 (RMS) error: {error:.6e}")
    print(f"Max error: {max_error:.6e}")
    print(f"NaN count: {np.isnan(u_grid).sum()}")
    print(f"Solver info: {result['solver_info']}")
