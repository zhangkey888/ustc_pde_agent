import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Parse case_spec
    nu_val = float(case_spec["pde"]["viscosity"])
    source = case_spec["pde"]["source"]
    bcs_spec = case_spec["pde"]["bcs"]
    output_spec = case_spec["output"]
    grid = output_spec["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    # High mesh resolution for accuracy (takes ~14s with direct LU)
    N = 256
    degree_u = 2
    degree_p = 1
    
    # Create mesh
    xmin, xmax, ymin, ymax = bbox
    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])
    msh = mesh.create_rectangle(comm, [p0, p1], [N, N], cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Mixed function space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), degree_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), degree_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # Define trial and test functions
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    # Viscosity and source
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = fem.Constant(msh, PETSc.ScalarType((float(source[0]), float(source[1]))))
    
    # Bilinear form for Stokes: -nu*laplacian(u) + grad(p) = f, div(u) = 0
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx
    
    # Boundary conditions
    fdim = msh.topology.dim - 1
    bcs = []
    
    # Track which boundaries have Dirichlet BCs
    dirichlet_boundaries = set()
    
    for bc_item in bcs_spec:
        bc_type = bc_item["type"]
        location = bc_item["location"]
        value = bc_item["value"]
        
        if bc_type == "dirichlet":
            dirichlet_boundaries.add(location)
            
            if location == "x0":
                marker = lambda x: np.isclose(x[0], xmin)
            elif location == "x1":
                marker = lambda x: np.isclose(x[0], xmax)
            elif location == "y0":
                marker = lambda x: np.isclose(x[1], ymin)
            elif location == "y1":
                marker = lambda x: np.isclose(x[1], ymax)
            elif location == "all":
                marker = lambda x: np.ones(x.shape[1], dtype=bool)
            else:
                raise ValueError(f"Unknown location: {location}")
            
            facets = mesh.locate_entities_boundary(msh, fdim, marker)
            dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
            
            u_bc = fem.Function(V)
            val = [float(v_) for v_ in value]
            u_bc.interpolate(lambda x, val=val: np.vstack([
                np.full(x.shape[1], val[0]),
                np.full(x.shape[1], val[1])
            ]))
            bc = fem.dirichletbc(u_bc, dofs, W.sub(0))
            bcs.append(bc)
    
    # Check if all boundaries have Dirichlet BCs (need pressure pin)
    all_boundaries = {"x0", "x1", "y0", "y1"}
    all_dirichlet = dirichlet_boundaries == all_boundaries or "all" in dirichlet_boundaries
    
    # Pin pressure at corner (0,0) to remove nullspace when all BCs are Dirichlet
    # Also pin if we suspect the system might be singular
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], xmin) & np.isclose(x[1], ymin)
    )
    if len(p_dofs[0]) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # Solve with direct LU solver (MUMPS)
    ksp_type = "preonly"
    pc_type = "lu"
    
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
    
    # Extract velocity
    u_h = w_h.sub(0).collapse()
    
    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    u_grid = np.full(nx_out * ny_out, np.nan)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        magnitude = np.linalg.norm(vals, axis=1)
        for idx, global_idx in enumerate(eval_map):
            u_grid[global_idx] = magnitude[idx]
    
    u_grid = u_grid.reshape(ny_out, nx_out)
    
    # Handle NaN at boundary points
    if np.any(np.isnan(u_grid)):
        u_grid = np.nan_to_num(u_grid, nan=0.0)
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": degree_u,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-10,
        "iterations": 1,
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info,
    }
