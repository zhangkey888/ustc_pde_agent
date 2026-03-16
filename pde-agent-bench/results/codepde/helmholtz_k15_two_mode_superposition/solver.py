import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    k_val = float(pde_config.get("params", {}).get("k", 15.0))
    
    # 2. Create mesh - use higher resolution for k=15 (need ~10 points per wavelength)
    # Wavelength ~ 2*pi/k ~ 0.42, so need mesh size ~ 0.04 or smaller
    # With degree 2, we can use fewer elements
    nx, ny = 80, 80
    degree = 2
    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    # Manufactured solution: u_exact = sin(2*pi*x)*sin(pi*y) + sin(pi*x)*sin(3*pi*y)
    u_exact = ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1]) + ufl.sin(pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    # -∇²u - k²u = f
    # For u1 = sin(2*pi*x)*sin(pi*y): -∇²u1 = (4*pi² + pi²)*u1 = 5*pi²*u1
    # For u2 = sin(pi*x)*sin(3*pi*y): -∇²u2 = (pi² + 9*pi²)*u2 = 10*pi²*u2
    # f = -∇²u - k²u = 5*pi²*u1 + 10*pi²*u2 - k²*(u1 + u2)
    # f = (5*pi² - k²)*u1 + (10*pi² - k²)*u2
    
    k2 = fem.Constant(domain, PETSc.ScalarType(k_val**2))
    
    u1 = ufl.sin(2 * pi * x[0]) * ufl.sin(pi * x[1])
    u2 = ufl.sin(pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    f_expr = (5.0 * pi**2 - k2) * u1 + (10.0 * pi**2 - k2) * u2
    
    # Weak form: ∫ ∇u·∇v dx - k² ∫ u*v dx = ∫ f*v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # 5. Boundary conditions - u = g = u_exact on ∂Ω
    # On the unit square boundary, sin(2*pi*x)*sin(pi*y) + sin(pi*x)*sin(3*pi*y)
    # At x=0 or x=1: sin(0)*... + sin(0 or pi)*... = 0 (since sin(2*pi)=0, sin(pi)=0)
    # At y=0 or y=1: ...*sin(0) + ...*sin(0 or 3*pi) = 0
    # So g = 0 on all boundaries
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # 6. Solve - use direct solver for robustness with indefinite Helmholtz
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    uh = problem.solve()
    
    # 7. Extract solution on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    
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
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 1,
        }
    }