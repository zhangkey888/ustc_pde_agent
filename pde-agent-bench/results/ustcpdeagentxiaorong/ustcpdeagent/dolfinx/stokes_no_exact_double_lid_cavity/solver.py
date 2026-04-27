import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve Stokes flow with Taylor-Hood elements."""
    
    nu_val = case_spec["pde"]["viscosity"]
    bcs_spec = case_spec["pde"]["bcs"]
    
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    N = 256
    
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    degree_u = 2
    degree_p = 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    bcs = []
    for bc_spec in bcs_spec:
        bc_value = bc_spec["value"]
        bc_where = bc_spec["where"]
        
        if bc_where == "x0":
            marker = lambda x: np.isclose(x[0], 0.0)
        elif bc_where == "x1":
            marker = lambda x: np.isclose(x[0], 1.0)
        elif bc_where == "y0":
            marker = lambda x: np.isclose(x[1], 0.0)
        elif bc_where == "y1":
            marker = lambda x: np.isclose(x[1], 1.0)
        else:
            raise ValueError(f"Unknown boundary: {bc_where}")
        
        facets = mesh.locate_entities_boundary(msh, fdim, marker)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        
        u_bc = fem.Function(V)
        bc_val = [float(bc_value[0]), float(bc_value[1])]
        u_bc.interpolate(lambda x, val=bc_val: np.vstack([
            np.full(x.shape[1], val[0]),
            np.full(x.shape[1], val[1])
        ]))
        
        bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
        bcs.append(bc)
    
    # Pressure pinning at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0 = fem.Function(Q)
    p0.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
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
    
    u_h = w_h.sub(0).collapse()
    
    # Sample velocity magnitude on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_grid = np.zeros(nx_out * ny_out)
    
    if len(points_on_proc) > 0:
        points_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        u_vals = u_h.eval(points_arr, cells_arr)
        magnitudes = np.sqrt(u_vals[:, 0]**2 + u_vals[:, 1]**2)
        for idx, global_idx in enumerate(eval_map):
            u_grid[global_idx] = magnitudes[idx]
    
    u_grid = u_grid.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": 1,
    }
    
    return {"u": u_grid, "solver_info": solver_info}
