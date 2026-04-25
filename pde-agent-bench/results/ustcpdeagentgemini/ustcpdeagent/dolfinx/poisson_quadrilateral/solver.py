import time
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc

def solve(case_spec: dict) -> dict:
    # 1. Parse parameters
    res = 128
    degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-9
    
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_spec.get("nx", 100)
    ny_out = grid_spec.get("ny", 100)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    comm = MPI.COMM_WORLD
    
    # 2. Create mesh
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [res, res], cell_type=mesh.CellType.quadrilateral)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Trial/Test functions
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # 5. Exact solution and source term
    x = ufl.SpatialCoordinate(domain)
    # u = exp(x)*cos(2*pi*y)
    u_exact = ufl.exp(x[0]) * ufl.cos(2 * ufl.pi * x[1])
    
    # -∇·(2.0 ∇u) = f
    # \Delta u = (1 - 4*pi^2)*u
    kappa = 2.0
    f = -kappa * (1.0 - 4.0 * ufl.pi**2) * u_exact
    
    # 6. Variational form
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 7. Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    
    # Locate all boundary facets
    def boundary_marker(x):
        return np.logical_or(np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0)),
                             np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
                             
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    dofs_bc = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, dofs_bc)
    
    # 8. Solve
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type, "ksp_rtol": rtol},
        petsc_options_prefix="pdebench_"
    )
    u_sol = problem.solve()
    
    # Try to get iteration count (might be 0 for LinearProblem if high level doesn't expose it easily)
    # We can just report a dummy or access the KSP if needed. Here we just set it to 1 since direct solve is fast anyway.
    iterations = problem.solver.getIterationNumber()
    
    # 9. Interpolate to grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_grid = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
        
    u_grid = u_grid.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": res,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }
