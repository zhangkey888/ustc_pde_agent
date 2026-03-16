import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", {})
    k_val = float(pde_config.get("wavenumber", 18.0))
    
    # For k=18, we need a fine mesh to resolve the oscillations
    # Rule of thumb: ~10 elements per wavelength, wavelength = 2*pi/k ≈ 0.349
    # On unit square, need N ~ 10*k/(2*pi) * some safety factor
    # Use higher order elements for efficiency
    
    element_degree = 2
    mesh_resolution = 120  # Fine enough for k=18 with P2 elements
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Source term: f = 12*(exp(-90*((x-0.3)**2 + (y-0.7)**2)) - exp(-90*((x-0.7)**2 + (y-0.3)**2)))
    f_expr = 12.0 * (
        ufl.exp(-90.0 * ((x[0] - 0.3)**2 + (x[1] - 0.7)**2))
        - ufl.exp(-90.0 * ((x[0] - 0.7)**2 + (x[1] - 0.3)**2))
    )
    
    k_const = fem.Constant(domain, PETSc.ScalarType(k_val))
    
    # Helmholtz: -∇²u - k²u = f
    # Weak form: ∫ ∇u·∇v dx - k² ∫ u·v dx = ∫ f·v dx
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_const**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # 5. Boundary conditions: u = 0 on ∂Ω (g=0, homogeneous Dirichlet)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)
    
    # 6. Solve - use direct solver for indefinite Helmholtz (most robust)
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
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
            "mesh_resolution": mesh_resolution,
            "element_degree": element_degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": 1,
        }
    }