import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # 1. Mesh setup
    nx_mesh = 64
    ny_mesh = 64
    element_degree = 3
    comm = MPI.COMM_WORLD
    
    domain = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    
    # 2. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 3. Exact solution for f and BC
    x = ufl.SpatialCoordinate(domain)
    # Exact solution: u = x*(1-x)*y*(1-y)*(1 + 0.5*x*y)
    u_ex = x[0] * (1 - x[0]) * x[1] * (1 - x[1]) * (1.0 + 0.5 * x[0] * x[1])
    
    f = -ufl.div(ufl.grad(u_ex))
    
    # 4. Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    def boundary_marker(x_pts):
        return np.logical_or.reduce([
            np.isclose(x_pts[0], 0.0),
            np.isclose(x_pts[0], 1.0),
            np.isclose(x_pts[1], 0.0),
            np.isclose(x_pts[1], 1.0)
        ])
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_ex, V.element.interpolation_points())
    u_bc.interpolate(u_bc_expr)
    
    bc = fem.dirichletbc(u_bc, bdofs)
    
    # 5. Variational form
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 6. Linear Problem solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol
        },
        petsc_options_prefix="poisson_"
    )
    
    u_sol = problem.solve()
    
    iterations = problem.solver.getIterationNumber()
    
    # 7. Interpolate onto target grid
    grid_spec = case_spec["output"]["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
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
            
    u_grid = np.full((nx_out * ny_out,), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
        
    u_grid = u_grid.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

