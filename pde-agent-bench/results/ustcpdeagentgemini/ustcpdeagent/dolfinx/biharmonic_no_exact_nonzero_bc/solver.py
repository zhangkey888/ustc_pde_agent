import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element, mixed_element
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # 1. Mesh setup
    nx_mesh = 128
    ny_mesh = 128
    msh = mesh.create_unit_square(MPI.COMM_WORLD, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)
    
    # 2. Function Spaces (Mixed Formulation)
    degree = 2
    # In DOLFINx 0.10.0, basix element creation syntax
    from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
    el = basix_element("Lagrange", msh.topology.cell_name(), degree)
    W_el = basix_mixed_element([el, el]) # Space for (u, v), where v = Delta u
    W = fem.functionspace(msh, W_el)
    
    # Subspaces for BCs
    V_u, _ = W.sub(0).collapse()
    V_v, _ = W.sub(1).collapse()
    
    # 3. Trial/Test functions
    (u, v) = ufl.TrialFunctions(W)
    (z, w) = ufl.TestFunctions(W)  # z tests u-equation, w tests v-equation
    
    # 4. Exact Solutions and Forcing
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    
    # u_exact = sin(3*pi*x) + cos(2*pi*y)
    u_exact = ufl.sin(3*pi*x[0]) + ufl.cos(2*pi*x[1])
    
    # v_exact = Delta u_exact = -9*pi^2 * sin(3*pi*x) - 4*pi^2 * cos(2*pi*y)
    v_exact = -9*pi**2 * ufl.sin(3*pi*x[0]) - 4*pi**2 * ufl.cos(2*pi*x[1])
    
    # f = Delta v_exact = Delta^2 u_exact = 81*pi^4 * sin(3*pi*x) + 16*pi^4 * cos(2*pi*y)
    f = 81*pi**4 * ufl.sin(3*pi*x[0]) + 16*pi**4 * ufl.cos(2*pi*x[1])
    
    # 5. Weak Form
    # Eq 1: Delta v = f  => -int(grad(v) . grad(w)) = int(f * w) => int(grad(v) . grad(w)) = -int(f * w)
    # Eq 2: Delta u = v  => -int(grad(u) . grad(z)) = int(v * z) => int(grad(u) . grad(z)) + int(v * z) = 0
    
    a = ufl.inner(ufl.grad(v), ufl.grad(w)) * ufl.dx \
      + ufl.inner(ufl.grad(u), ufl.grad(z)) * ufl.dx \
      + ufl.inner(v, z) * ufl.dx
      
    L = -f * w * ufl.dx
    
    # 6. Boundary Conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    # BC for u
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_u), fdim, boundary_facets)
    u_bc_func = fem.Function(V_u)
    u_bc_func.interpolate(fem.Expression(u_exact, V_u.element.interpolation_points()))
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # BC for v
    dofs_v = fem.locate_dofs_topological((W.sub(1), V_v), fdim, boundary_facets)
    v_bc_func = fem.Function(V_v)
    v_bc_func.interpolate(fem.Expression(v_exact, V_v.element.interpolation_points()))
    bc_v = fem.dirichletbc(v_bc_func, dofs_v, W.sub(1))
    
    bcs = [bc_u, bc_v]
    
    # 7. Solve Linear System
    ksp_type = "preonly"
    pc_type = "lu"
    problem = petsc.LinearProblem(a, L, bcs=bcs,
                                  petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
                                  petsc_options_prefix="biharm_")
    w_sol = problem.solve()
    
    u_sol = w_sol.sub(0).collapse()
    
    # 8. Sample on output grid
    grid_spec = case_spec["output"]["grid"]
    nx = grid_spec["nx"]
    ny = grid_spec["ny"]
    bbox = grid_spec["bbox"]
    
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack((XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())))
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
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
            
    u_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
        
    u_grid = u_values.reshape((ny, nx))
    
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {
        "u": u_grid,
        "solver_info": solver_info
    }

