import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    # Parse case spec
    pde = case_spec["pde"]
    k_val = pde["wavenumber"]["k"]
    
    # Source term
    f_val = pde["source_term"]  # 0.0 for this case
    
    # Output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]  # [xmin, xmax, ymin, ymax]
    
    # Solver parameters
    # k=12, wavelength ~ 2*pi/12 ~ 0.524
    # P4 elements with N=120 gives extremely high accuracy
    degree = 4
    N = 120

    comm = MPI.COMM_WORLD
    
    # Create mesh
    xmin, xmax, ymin, ymax = bbox
    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    
    # Function space
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Define variational problem
    # -∇²u - k²u = f  =>  ∫ grad(u)·grad(v) dx - k² ∫ u·v dx = ∫ f·v dx
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    k2 = fem.Constant(domain, ScalarType(k_val**2))
    f_const = fem.Constant(domain, ScalarType(float(f_val)))
    
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k2 * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f_const, v) * ufl.dx
    
    # Boundary conditions
    # u = sin(3*pi*x) + cos(2*pi*y) on all boundary
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Create BC function using UFL expression
    u_bc = fem.Function(V)
    x_coord = ufl.SpatialCoordinate(domain)
    bc_expr = ufl.sin(3 * ufl.pi * x_coord[0]) + ufl.cos(2 * ufl.pi * x_coord[1])
    expr = fem.Expression(bc_expr, V.element.interpolation_points)
    u_bc.interpolate(expr)
    
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Solve using direct LU (best for indefinite Helmholtz)
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
        petsc_options_prefix="helmholtz_"
    )
    u_sol = problem.solve()
    
    # Sample solution onto output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((3, nx_out * ny_out))
    pts[0, :] = XX.ravel()
    pts[1, :] = YY.ravel()
    
    # Build bounding box tree and find cells
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape(ny_out, nx_out)
    
    # Solver info
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": 1,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
