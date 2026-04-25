import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # 1. Mesh and Function Space
    mesh_res = 128
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    element_deg = 2
    V = fem.functionspace(domain, ("Lagrange", element_deg))
    
    # 2. Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 3. Variational Form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    # Source term: f = exp(-200*((x-0.25)**2 + (y-0.75)**2))
    f_expr = ufl.exp(-200.0 * ((x[0] - 0.25)**2 + (x[1] - 0.75)**2))
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # 4. Solver Setup
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-8},
        petsc_options_prefix="poisson_"
    )
    
    # 5. Solve
    u_sol = problem.solve()
    
    # Get iterations from the underlying KSP solver
    ksp = problem.solver
    iters = ksp.getIterationNumber()
    
    # 6. Evaluation on grid
    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = bb_tree(domain, domain.topology.dim)
    cell_candidates = compute_collisions_points(tree, pts)
    colliding_cells = compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_grid_flat = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
        
    u_grid = u_grid_flat.reshape((ny_out, nx_out))
    
    # Check if there are any NaNs and replace with 0.0 or nearest (for parallel / boundaries)
    # Since we evaluate on domain, there shouldn't be NaNs unless points are exactly outside.
    u_grid = np.nan_to_num(u_grid)
    
    # Ensure parallel combination if running with MPI (though typically tested in serial)
    if comm.size > 1:
        u_grid_global = np.zeros_like(u_grid)
        comm.Reduce(u_grid, u_grid_global, op=MPI.MAX, root=0)
        u_grid = u_grid_global
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": element_deg,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-8,
            "iterations": iters
        }
    }
