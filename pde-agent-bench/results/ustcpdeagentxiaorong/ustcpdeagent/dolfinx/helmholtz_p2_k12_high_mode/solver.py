import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Extract parameters from case_spec
    k_val = case_spec.get("pde", {}).get("parameters", {}).get("k", 12.0)
    
    # Output grid specification
    grid_info = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_info.get("nx", 100)
    ny_out = grid_info.get("ny", 100)
    bbox = grid_info.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    element_degree = 3
    mesh_resolution = 80
    
    # Create mesh
    domain = mesh.create_rectangle(
        comm,
        [np.array([bbox[0], bbox[2]]), np.array([bbox[1], bbox[3]])],
        [mesh_resolution, mesh_resolution],
        cell_type=mesh.CellType.triangle
    )
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # Define the exact solution and source term
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    
    u_exact = ufl.sin(3 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    # Source term: f = -∇²u - k²u
    # For u = sin(3πx)sin(3πy):
    # ∇²u = -18π²sin(3πx)sin(3πy)
    # f = 18π²sin(3πx)sin(3πy) - k²sin(3πx)sin(3πy) = (18π² - k²)sin(3πx)sin(3πy)
    k_const = fem.Constant(domain, ScalarType(k_val))
    f_expr = (18.0 * pi**2 - k_const**2) * ufl.sin(3 * pi * x[0]) * ufl.sin(3 * pi * x[1])
    
    # Dirichlet BCs: u = g on ∂Ω (= 0 for this manufactured solution on [0,1]²)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Variational form: ∫ ∇u·∇v dx - k² ∫ u·v dx = ∫ f·v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k_const**2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx
    
    # Solve with direct solver (LU) for indefinite system
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-12
    
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Sample solution onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    # Build bounding box tree
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    
    # Find cells for each point
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_grid[eval_map] = vals.flatten()
    
    u_grid = u_grid.reshape(ny_out, nx_out)
    
    # Get iteration count
    try:
        iterations = problem.solver.getIterationNumber()
    except:
        iterations = 1
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
