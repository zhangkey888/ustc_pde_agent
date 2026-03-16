import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    k_val = float(pde_config.get("wavenumber", 5.0))
    
    # 2. Create mesh
    nx, ny = 80, 80
    degree = 2
    domain = mesh.create_unit_square(MPI.COMM_WORLD, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u_exact = sin(pi*x)*sin(pi*y)
    # -∇²u - k²u = f
    # -∇²(sin(pi*x)*sin(pi*y)) = 2*pi²*sin(pi*x)*sin(pi*y)
    # f = 2*pi²*sin(pi*x)*sin(pi*y) - k²*sin(pi*x)*sin(pi*y)
    # f = (2*pi² - k²)*sin(pi*x)*sin(pi*y)
    
    pi = ufl.pi
    u_exact_expr = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    f_expr = (2.0 * pi**2 - k_val**2) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
    
    # Weak form: ∫ ∇u·∇v dx - k² ∫ u*v dx = ∫ f*v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_val**2 * ufl.inner(u, v) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # 5. Boundary conditions (u = sin(pi*x)*sin(pi*y) = 0 on boundary of unit square)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # 6. Solve - use GMRES with ILU for indefinite Helmholtz
    ksp_type = "gmres"
    pc_type = "ilu"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
            "ksp_max_it": "2000",
        },
        petsc_options_prefix="helmholtz_"
    )
    uh = problem.solve()
    
    # 7. Extract solution on 50x50 uniform grid
    ngrid = 50
    xs = np.linspace(0.0, 1.0, ngrid)
    ys = np.linspace(0.0, 1.0, ngrid)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.zeros((3, ngrid * ngrid))
    points[0, :] = X.ravel()
    points[1, :] = Y.ravel()
    points[2, :] = 0.0
    
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
    
    u_values = np.full(ngrid * ngrid, np.nan)
    if len(points_on_proc) > 0:
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((ngrid, ngrid))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": -1,
        }
    }