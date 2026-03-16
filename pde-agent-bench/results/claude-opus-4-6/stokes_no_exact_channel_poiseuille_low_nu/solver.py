import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, nls
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parameters
    nu_val = 0.05
    mesh_resolution = 80
    degree_u = 2
    degree_p = 1
    
    # Create mesh
    domain = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    # Function spaces (Taylor-Hood P2/P1)
    V = fem.functionspace(domain, ("Lagrange", degree_u, (domain.geometry.dim,)))
    Q = fem.functionspace(domain, ("Lagrange", degree_p))
    
    # Mixed function space
    mel = ufl.MixedElement([V.ufl_element(), Q.ufl_element()])
    W = fem.functionspace(domain, mel)
    
    # Trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Viscosity
    nu = fem.Constant(domain, ScalarType(nu_val))
    
    # Source term
    f = fem.Constant(domain, ScalarType((0.0, 0.0)))
    
    # Bilinear form for Stokes
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + q * ufl.div(u) * ufl.dx)
    
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions - Poiseuille channel flow
    # For a channel flow in the x-direction on [0,1]x[0,1]:
    # u_x = 4*y*(1-y) on inlet (x=0), u = (0,0) on walls (y=0, y=1)
    # u_y = 0 everywhere on boundary
    # On outlet (x=1), we can set the same parabolic profile or use do-nothing
    # For Poiseuille: u = (4*y*(1-y), 0) on inlet/outlet, u=(0,0) on walls
    
    # Inlet: x = 0
    def inlet(x):
        return np.isclose(x[0], 0.0)
    
    # Outlet: x = 1
    def outlet(x):
        return np.isclose(x[0], 1.0)
    
    # Bottom wall: y = 0
    def bottom(x):
        return np.isclose(x[1], 0.0)
    
    # Top wall: y = 1
    def top(x):
        return np.isclose(x[1], 1.0)
    
    # Create BC functions
    # Parabolic inlet profile
    u_inlet = fem.Function(V)
    u_inlet.interpolate(lambda x: np.vstack([4.0 * x[1] * (1.0 - x[1]), np.zeros_like(x[0])]))
    
    # No-slip
    u_noslip = fem.Function(V)
    u_noslip.interpolate(lambda x: np.vstack([np.zeros_like(x[0]), np.zeros_like(x[0])]))
    
    # Locate facets
    inlet_facets = mesh.locate_entities_boundary(domain, fdim, inlet)
    outlet_facets = mesh.locate_entities_boundary(domain, fdim, outlet)
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
    top_facets = mesh.locate_entities_boundary(domain, fdim, top)
    
    # Locate DOFs
    V_sub, _ = W.sub(0).collapse()
    
    dofs_inlet = fem.locate_dofs_topological((W.sub(0), V), fdim, inlet_facets)
    dofs_outlet = fem.locate_dofs_topological((W.sub(0), V), fdim, outlet_facets)
    dofs_bottom = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    dofs_top = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    
    bc_inlet = fem.dirichletbc(u_inlet, dofs_inlet, W.sub(0))
    bc_outlet = fem.dirichletbc(u_inlet, dofs_outlet, W.sub(0))
    bc_bottom = fem.dirichletbc(u_noslip, dofs_bottom, W.sub(0))
    bc_top = fem.dirichletbc(u_noslip, dofs_top, W.sub(0))
    
    bcs = [bc_inlet, bc_outlet, bc_bottom, bc_top]
    
    # Solve
    ksp_type = "gmres"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_rtol": str(rtol),
        },
        petsc_options_prefix="stokes_"
    )
    wh = problem.solve()
    wh.x.scatter_forward()
    
    # Extract velocity
    u_sol = wh.sub(0).collapse()
    
    # Sample on 100x100 grid
    nx_out, ny_out = 100, 100
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points = np.zeros((3, nx_out * ny_out))
    points[0, :] = XX.ravel()
    points[1, :] = YY.ravel()
    
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
    
    vel_magnitude = np.full(nx_out * ny_out, np.nan)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        # vals shape: (n_points, 2) for 2D velocity
        mag = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            vel_magnitude[global_idx] = mag[idx]
    
    u_grid = vel_magnitude.reshape((nx_out, ny_out))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": rtol,
        "iterations": 1,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }