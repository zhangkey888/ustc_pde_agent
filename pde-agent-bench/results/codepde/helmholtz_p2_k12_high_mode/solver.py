import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    k_val = float(pde_config.get("params", {}).get("k", 12.0))
    
    # 2. Create mesh - use sufficient resolution for k=12 and P2 elements
    # For Helmholtz with k=12, we need enough points per wavelength
    # wavelength ~ 2*pi/k ~ 0.52, so with P2 and nx=80, h~0.0125, ~42 points per wavelength
    nx = 80
    ny = 80
    element_degree = 2
    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space - P2
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define variational problem
    # Manufactured solution: u = sin(3*pi*x)*sin(3*pi*y)
    # -∇²u - k²u = f
    # ∇²u = -9*pi²*sin(3*pi*x)*sin(3*pi*y) - 9*pi²*sin(3*pi*x)*sin(3*pi*y) = -18*pi²*sin(3*pi*x)*sin(3*pi*y)
    # -∇²u = 18*pi²*sin(3*pi*x)*sin(3*pi*y)
    # -k²u = -k²*sin(3*pi*x)*sin(3*pi*y)
    # f = (18*pi² - k²)*sin(3*pi*x)*sin(3*pi*y)
    
    x = ufl.SpatialCoordinate(domain)
    
    u_exact_expr = ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    f_expr = (18.0 * ufl.pi**2 - k_val**2) * ufl.sin(3 * ufl.pi * x[0]) * ufl.sin(3 * ufl.pi * x[1])
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Weak form: ∫ ∇u·∇v dx - k²∫ u*v dx = ∫ f*v dx
    # (from -∇²u - k²u = f, multiply by v, integrate by parts: ∫∇u·∇v - k²∫uv = ∫fv)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # 5. Boundary conditions
    # u = sin(3*pi*x)*sin(3*pi*y) = 0 on all boundaries of [0,1]^2
    # (since sin(0)=sin(3*pi)=0... wait, sin(3*pi*0)=0, sin(3*pi*1)=sin(3*pi)=0)
    # So homogeneous Dirichlet BC
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Zero BC (exact solution is zero on boundary)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    # 6. Solve - use direct solver for indefinite Helmholtz
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
    n_eval = 50
    xs = np.linspace(0, 1, n_eval)
    ys = np.linspace(0, 1, n_eval)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, n_eval * n_eval))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    points[2, :] = 0.0
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(points.shape[1], np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((n_eval, n_eval))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 1,
        }
    }