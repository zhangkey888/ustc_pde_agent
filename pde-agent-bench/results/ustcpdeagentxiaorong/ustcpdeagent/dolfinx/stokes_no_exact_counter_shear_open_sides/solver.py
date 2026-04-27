import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1"

from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    """Solve Stokes flow with Taylor-Hood P2/P1 mixed elements."""
    
    nu_val = case_spec["pde"]["viscosity"]
    source = case_spec["pde"]["source"]
    bcs_spec = case_spec["pde"]["bcs"]
    
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    N = 64
    degree_u = 2
    degree_p = 1
    
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Taylor-Hood P2/P1 mixed space
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f_val = [float(s) for s in source]
    f = fem.Constant(msh, PETSc.ScalarType(np.array(f_val)))
    
    # Stokes weak form
    a_form = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
              - p * ufl.div(v) * ufl.dx
              + ufl.div(u) * q * ufl.dx)
    L_form = ufl.inner(f, v) * ufl.dx
    
    # Parse and apply Dirichlet BCs
    bcs = []
    bc_locations = set()
    
    for bc_spec in bcs_spec:
        if bc_spec["type"] != "dirichlet":
            continue
        location = bc_spec["location"]
        value = bc_spec["value"]
        bc_locations.add(location)
        
        if location == "y1":
            marker = lambda x: np.isclose(x[1], 1.0)
        elif location == "y0":
            marker = lambda x: np.isclose(x[1], 0.0)
        elif location == "x0":
            marker = lambda x: np.isclose(x[0], 0.0)
        elif location == "x1":
            marker = lambda x: np.isclose(x[0], 1.0)
        else:
            continue
        
        facets = mesh.locate_entities_boundary(msh, fdim, marker)
        dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
        u_bc = fem.Function(V)
        val = np.array([float(v_) for v_ in value])
        u_bc.interpolate(lambda x, val=val: np.outer(val, np.ones(x.shape[1])))
        bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
        bcs.append(bc)
    
    # If all 4 boundaries have velocity Dirichlet BCs, pin pressure
    all_walls = {"x0", "x1", "y0", "y1"}
    if bc_locations >= all_walls:
        p_dofs = fem.locate_dofs_geometrical(
            (W.sub(1), Q),
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
        )
        if len(p_dofs[0]) > 0:
            p0_func = fem.Function(Q)
            p0_func.x.array[:] = 0.0
            bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
            bcs.append(bc_p)
    
    # Solve with MUMPS direct solver
    problem = petsc.LinearProblem(
        a_form, L_form, bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="stokes_"
    )
    
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()
    
    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((XX.size, 3))
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
    
    u_values = np.full((len(pts), gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for idx, global_idx in enumerate(eval_map):
            u_values[global_idx, :] = vals[idx, :]
    
    vel_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = vel_mag.reshape(ny_out, nx_out)
    
    # Handle NaN at boundary points via nearest neighbor interpolation
    if np.any(np.isnan(u_grid)):
        from scipy.interpolate import NearestNDInterpolator
        valid = ~np.isnan(u_grid.ravel())
        if np.any(valid):
            interp = NearestNDInterpolator(
                np.column_stack([XX.ravel()[valid], YY.ravel()[valid]]),
                u_grid.ravel()[valid]
            )
            nan_mask = np.isnan(u_grid.ravel())
            u_grid_flat = u_grid.ravel()
            u_grid_flat[nan_mask] = interp(XX.ravel()[nan_mask], YY.ravel()[nan_mask])
            u_grid = u_grid_flat.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": 1,
        },
    }
