import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc
import basix


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case_spec
    nu_val = 1.0
    
    # Mesh resolution - use high resolution for accuracy
    N = 64
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    
    # Taylor-Hood mixed elements: P2/P1
    P2 = fem.functionspace(domain, ("Lagrange", 2, (domain.geometry.dim,)))
    P1 = fem.functionspace(domain, ("Lagrange", 1))
    
    # Mixed function space via MixedElement approach
    vel_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim,))
    pres_el = basix.ufl.element("Lagrange", domain.topology.cell_name(), 1)
    mixed_el = basix.ufl.mixed_element([vel_el, pres_el])
    
    W = fem.functionspace(domain, mixed_el)
    
    # Define trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution
    # u_exact = [pi*exp(x)*cos(pi*y), -exp(x)*sin(pi*y)]
    # p_exact = exp(x)*cos(pi*y)
    pi_ = ufl.pi
    
    u_exact = ufl.as_vector([
        pi_ * ufl.exp(x[0]) * ufl.cos(pi_ * x[1]),
        -ufl.exp(x[0]) * ufl.sin(pi_ * x[1])
    ])
    p_exact = ufl.exp(x[0]) * ufl.cos(pi_ * x[1])
    
    # Compute source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    # For component u1 = pi*exp(x)*cos(pi*y):
    #   d^2u1/dx^2 = pi*exp(x)*cos(pi*y)
    #   d^2u1/dy^2 = -pi^3*exp(x)*cos(pi*y)
    #   laplacian(u1) = pi*exp(x)*cos(pi*y) - pi^3*exp(x)*cos(pi*y) = pi*(1-pi^2)*exp(x)*cos(pi*y)
    # For component u2 = -exp(x)*sin(pi*y):
    #   d^2u2/dx^2 = -exp(x)*sin(pi*y)
    #   d^2u2/dy^2 = pi^2*exp(x)*sin(pi*y)
    #   laplacian(u2) = -exp(x)*sin(pi*y) + pi^2*exp(x)*sin(pi*y) = (pi^2-1)*exp(x)*sin(pi*y)
    # grad(p_exact) = [exp(x)*cos(pi*y), -pi*exp(x)*sin(pi*y)]
    # f = -nu*laplacian(u_exact) + grad(p_exact)
    
    nu = fem.Constant(domain, PETSc.ScalarType(nu_val))
    
    # Use UFL to compute the source term symbolically
    f = -nu * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # Bilinear form for Stokes
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)
    
    # Linear form
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions: u = u_exact on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Find all boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Velocity BC
    W0, W0_to_W = W.sub(0).collapse()
    
    u_bc_func = fem.Function(W0)
    u_exact_expr = fem.Expression(u_exact, W0.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), W0), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # Pin pressure at one point to fix the constant
    # Find a DOF near (0, 0) for pressure
    W1, W1_to_W = W.sub(1).collapse()
    
    p_bc_func = fem.Function(W1)
    p_exact_expr = fem.Expression(p_exact, W1.element.interpolation_points)
    p_bc_func.interpolate(p_exact_expr)
    
    # Pin pressure at a corner point
    def corner(x):
        return np.logical_and(np.isclose(x[0], 0.0), np.isclose(x[1], 0.0))
    
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W.sub(1), W1), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W.sub(1))
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]
    
    # Solve
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = petsc.create_vector(L_form)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)
    
    wh = fem.Function(W)
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.MINRES)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    solver.setTolerances(rtol=1e-12, atol=1e-14, max_it=5000)
    solver.setUp()
    
    solver.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    
    iterations = solver.getIterationNumber()
    
    # Extract velocity on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    # Extract the velocity subfunction
    uh_vel = wh.sub(0).collapse()
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[:, i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    vel_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh_vel.eval(pts_arr, cells_arr)
        for idx, global_idx in enumerate(eval_map):
            vel_values[global_idx, :] = vals[idx, :]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(vel_values[:, 0]**2 + vel_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "minres",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": iterations,
        }
    }