import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid spec
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    xmin, xmax, ymin, ymax = bbox
    
    # Discretization
    mesh_res = 128
    degree = 2
    domain = mesh.create_rectangle(comm, [[xmin, ymin], [xmax, ymax]], [mesh_res, mesh_res], cell_type=mesh.CellType.quadrilateral)
    
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Problem definition
    k = 10.0
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution and source
    u_ex = ufl.sin(2 * ufl.pi * x[0]) * ufl.cos(3 * ufl.pi * x[1])
    f_expr = (13 * ufl.pi**2 - k**2) * u_ex
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs_bc = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc_func, dofs_bc)
    
    # Weak form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Solve
    ksp_type = "preonly"
    pc_type = "lu"
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        petsc_options_prefix="helmholtz_"
    )
    
    u_sol = problem.solve()
    
    # Interpolation on uniform grid
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-8,
        "iterations": 1,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }
