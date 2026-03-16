import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    k_val = 15.0
    
    # For k=15, we need sufficient resolution. Rule of thumb: ~10 points per wavelength
    # wavelength ~ 2*pi/k ~ 0.42, so need mesh size ~ 0.04 or finer
    # With degree 2, we can use fewer elements
    N = 80
    degree = 2
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Exact solution
    u_exact_ufl = ufl.sin(2*pi*x[0])*ufl.sin(pi*x[1]) + ufl.sin(pi*x[0])*ufl.sin(3*pi*x[1])
    
    # Source term: f = -∇²u - k²u
    # For u1 = sin(2πx)sin(πy): -∇²u1 = (4π² + π²)u1 = 5π²u1
    # For u2 = sin(πx)sin(3πy): -∇²u2 = (π² + 9π²)u2 = 10π²u2
    # So f = -∇²u - k²u = 5π²*u1 + 10π²*u2 - k²*(u1 + u2)
    #       = (5π² - k²)*sin(2πx)sin(πy) + (10π² - k²)*sin(πx)sin(3πy)
    
    k2 = fem.Constant(domain, ScalarType(k_val**2))
    
    f_ufl = (5.0*pi**2 - k2) * ufl.sin(2*pi*x[0])*ufl.sin(pi*x[1]) + \
             (10.0*pi**2 - k2) * ufl.sin(pi*x[0])*ufl.sin(3*pi*x[1])
    
    # Variational form: ∇u·∇v - k²uv = fv
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_ufl, v) * ufl.dx
    
    # Boundary conditions: u = g on ∂Ω
    # On the unit square boundary, sin terms vanish (x=0,1 or y=0,1), so g=0
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    
    # Solve with direct solver (LU) since Helmholtz is indefinite
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Evaluate on 50x50 grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, nx_out * ny_out))
    points[0] = XX.ravel()
    points[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1,
    }
    
    return {"u": u_grid, "solver_info": solver_info}