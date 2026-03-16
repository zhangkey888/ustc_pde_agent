import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    k_val = float(pde_config.get("helmholtz_k", 24.0))
    
    # For k=24, we need sufficient resolution. Rule of thumb: ~10 points per wavelength
    # wavelength = 2*pi/k ≈ 0.26, so need mesh size h ≈ 0.026 or smaller
    # With P2 elements we can use fewer cells
    N = 80  # mesh resolution
    degree = 2  # polynomial degree
    
    # 2. Create mesh
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Manufactured solution: u_exact = sin(5*pi*x)*sin(4*pi*y)
    u_exact = ufl.sin(5 * pi * x[0]) * ufl.sin(4 * pi * x[1])
    
    # -∇²u - k²u = f
    # For u_exact = sin(5*pi*x)*sin(4*pi*y):
    # -∇²u_exact = (25*pi² + 16*pi²) * sin(5*pi*x)*sin(4*pi*y) = 41*pi² * u_exact
    # So f = 41*pi² * u_exact - k² * u_exact = (41*pi² - k²) * u_exact
    
    k2 = fem.Constant(domain, PETSc.ScalarType(k_val**2))
    
    # Source term derived from manufactured solution
    f = (25.0 * pi**2 + 16.0 * pi**2) * ufl.sin(5 * pi * x[0]) * ufl.sin(4 * pi * x[1]) \
        - k2 * ufl.sin(5 * pi * x[0]) * ufl.sin(4 * pi * x[1])
    
    # Bilinear form: ∫ ∇u·∇v dx - k² ∫ u*v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx
    
    # 5. Boundary conditions
    # u = g = u_exact on ∂Ω
    # For this manufactured solution, u_exact = sin(5*pi*x)*sin(4*pi*y) = 0 on all boundaries
    # (since sin(0)=0 and sin(n*pi)=0 for integer n, and x,y ∈ {0,1})
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # 6. Solve using direct LU solver (robust for indefinite Helmholtz)
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="helmholtz_"
    )
    uh = problem.solve()
    
    # 7. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, 0] = points_2d[:, 0]
    points_3d[:, 1] = points_2d[:, 1]
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": 1,
        }
    }