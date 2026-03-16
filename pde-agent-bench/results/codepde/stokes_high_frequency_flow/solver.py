import numpy as np
from dolfinx import mesh, fem, default_scalar_type, geometry
from dolfinx.fem import petsc
from mpi4py import MPI
import ufl
from petsc4py import PETSc
import basix


def solve(case_spec: dict) -> dict:
    # 1. Parse case_spec
    pde_config = case_spec.get("pde", case_spec.get("oracle_config", {}).get("pde", {}))
    nu_val = float(pde_config.get("viscosity", 1.0))
    
    # 2. Create mesh - use high resolution for high-frequency solution
    N = 64
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    
    # Taylor-Hood elements: P2/P1
    degree_u = 2
    degree_p = 1
    
    # 3. Create mixed function space
    V_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_u, shape=(domain.geometry.dim,))
    Q_el = basix.ufl.element("Lagrange", domain.basix_cell(), degree_p)
    mixed_el = basix.ufl.mixed_element([V_el, Q_el])
    W = fem.functionspace(domain, mixed_el)
    
    # Also create individual spaces for BCs
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # 4. Define trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Spatial coordinates
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution
    # u_exact = [2*pi*cos(2*pi*y)*sin(2*pi*x), -2*pi*cos(2*pi*x)*sin(2*pi*y)]
    # p_exact = sin(2*pi*x)*cos(2*pi*y)
    pi = ufl.pi
    u_exact = ufl.as_vector([
        2*pi*ufl.cos(2*pi*x[1])*ufl.sin(2*pi*x[0]),
        -2*pi*ufl.cos(2*pi*x[0])*ufl.sin(2*pi*x[1])
    ])
    p_exact = ufl.sin(2*pi*x[0])*ufl.cos(2*pi*x[1])
    
    # Compute source term: f = -nu * laplacian(u_exact) + grad(p_exact)
    # For vector Laplacian, we use div(grad(u_exact))
    nu_c = fem.Constant(domain, default_scalar_type(nu_val))
    
    f = -nu_c * ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # 5. Variational form for Stokes
    a = (nu_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)
    
    L = ufl.inner(f, v) * ufl.dx
    
    # 6. Boundary conditions - Dirichlet for velocity on all boundaries
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # All boundary facets
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    
    # Interpolate exact velocity onto V
    u_bc_func = fem.Function(V)
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc_func.interpolate(u_exact_expr)
    
    # Locate DOFs for velocity sub-space
    W0 = W.sub(0)
    dofs_u = fem.locate_dofs_topological((W0, V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W0)
    
    # Pin pressure at one point to fix the constant
    # Find a DOF near (0,0) for pressure
    def corner(x):
        return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    
    W1 = W.sub(1)
    
    # Create pressure BC function
    p_bc_func = fem.Function(Q)
    p_exact_expr = fem.Expression(p_exact, Q.element.interpolation_points)
    p_bc_func.interpolate(p_exact_expr)
    
    # Pin pressure at corner
    corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
    if len(corner_facets) > 0:
        dofs_p = fem.locate_dofs_topological((W1, Q), fdim, corner_facets)
        bc_p = fem.dirichletbc(p_bc_func, dofs_p, W1)
        bcs = [bc_u, bc_p]
    else:
        bcs = [bc_u]
    
    # 7. Solve
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
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.setTolerances(rtol=1e-12, atol=1e-14, max_it=5000)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    
    solver.solve(b, wh.x.petsc_vec)
    wh.x.scatter_forward()
    iterations = solver.getIterationNumber()
    
    # 8. Extract velocity and compute magnitude on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
    # Get velocity sub-function
    uh = wh.sub(0).collapse()
    
    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
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
    
    u_values = np.full((points.shape[1], 2), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = uh.eval(pts_arr, cells_arr)
        for idx_local, idx_global in enumerate(eval_map):
            u_values[idx_global, :] = vals[idx_local, :]
    
    # Compute velocity magnitude
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape((nx_out, ny_out))
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "gmres",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": iterations,
        }
    }