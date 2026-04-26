import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case spec
    pde = case_spec["pde"]
    nu = float(pde["parameters"]["viscosity"])
    
    output_spec = case_spec["output"]
    grid_spec = output_spec["grid"]
    nx_out = grid_spec["nx"]
    ny_out = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    # Mesh resolution
    N = 384
    
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Taylor-Hood: P2 velocity, P1 pressure
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    a = (2.0 * nu * ufl.inner(eps(u), eps(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    bcs = []
    
    # Left (x=0): inflow
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    left_facets = mesh.locate_entities_boundary(msh, fdim, left_boundary)
    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    u_inflow = fem.Function(V)
    u_inflow.interpolate(lambda x: np.vstack([
        np.sin(np.pi * x[1]),
        0.2 * np.sin(2.0 * np.pi * x[1])
    ]))
    bc_left = fem.dirichletbc(u_inflow, left_dofs, W.sub(0))
    bcs.append(bc_left)
    
    # Bottom (y=0): no-slip
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, bottom_boundary)
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_noslip = fem.Function(V)
    u_noslip.x.array[:] = 0.0
    bc_bottom = fem.dirichletbc(u_noslip, bottom_dofs, W.sub(0))
    bcs.append(bc_bottom)
    
    # Top (y=1): no-slip
    def top_boundary(x):
        return np.isclose(x[1], 1.0)
    top_facets = mesh.locate_entities_boundary(msh, fdim, top_boundary)
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_noslip_top = fem.Function(V)
    u_noslip_top.x.array[:] = 0.0
    bc_top = fem.dirichletbc(u_noslip_top, top_dofs, W.sub(0))
    bcs.append(bc_top)
    
    # Pressure pin at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    problem = petsc.LinearProblem(
        a, L, bcs=bcs,
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    
    w_h = problem.solve()
    w_h.x.scatter_forward()
    
    u_h = w_h.sub(0).collapse()
    
    iterations = 1
    
    # Sample on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    pts = np.zeros((3, nx_out * ny_out))
    pts[0] = XX.ravel()
    pts[1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.zeros((ny_out, nx_out))
    if len(points_on_proc) > 0:
        u_vals = u_h.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_mag = np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)
        u_flat = np.zeros(nx_out * ny_out)
        u_flat[eval_map] = u_mag
        u_grid = u_flat.reshape(ny_out, nx_out)
    
    if comm.size > 1:
        u_grid_local = u_grid.copy()
        u_grid_global = np.zeros_like(u_grid)
        comm.Allreduce(u_grid_local, u_grid_global, op=MPI.SUM)
        u_grid = u_grid_global
    
    div_form = fem.form(ufl.div(u_h) * ufl.dx)
    div_integral = fem.assemble_scalar(div_form)
    div_integral = msh.comm.allreduce(div_integral, op=MPI.SUM)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": 2,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "mass_conservation_error": float(abs(div_integral)),
    }
    
    return {"u": u_grid, "solver_info": solver_info}
