import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    
    # High frequency in x (8*pi) requires fine mesh in x direction
    # Error target: 2.02e-03
    # For sin(8*pi*x)*sin(pi*y), we need sufficient resolution to capture 8 half-waves in x
    # With degree 2 elements, we need fewer elements than degree 1
    
    nx = 128
    ny = 32  # y-direction only has frequency pi, so fewer elements needed
    degree = 2
    
    # 2. Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    # 3. Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # 4. Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution: u_exact = sin(8*pi*x)*sin(pi*y)
    # kappa = 1.0
    # -div(kappa * grad(u)) = f
    # -( d^2u/dx^2 + d^2u/dy^2 ) = f
    # d^2u/dx^2 = -(8*pi)^2 * sin(8*pi*x)*sin(pi*y) = -64*pi^2 * sin(8*pi*x)*sin(pi*y)
    # d^2u/dy^2 = -(pi)^2 * sin(8*pi*x)*sin(pi*y)
    # f = (64*pi^2 + pi^2) * sin(8*pi*x)*sin(pi*y) = 65*pi^2 * sin(8*pi*x)*sin(pi*y)
    
    kappa = fem.Constant(domain, default_scalar_type(1.0))
    pi = ufl.pi
    
    u_exact = ufl.sin(8 * pi * x[0]) * ufl.sin(pi * x[1])
    f_expr = 65.0 * pi**2 * ufl.sin(8 * pi * x[0]) * ufl.sin(pi * x[1])
    
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx
    
    # 5. Boundary conditions
    # u = g = sin(8*pi*x)*sin(pi*y) on boundary
    # On all boundaries of unit square, sin(pi*y)=0 at y=0,1 and sin(8*pi*x)=0 at x=0,1
    # So g = 0 on all boundaries
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)
    
    # 6. Solve
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10
    
    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": str(rtol),
        },
        petsc_options_prefix="poisson_"
    )
    uh = problem.solve()
    
    # 7. Extract on 50x50 uniform grid
    nx_out, ny_out = 50, 50
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
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
        vals = uh.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": max(nx, ny),
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": -1,
        }
    }