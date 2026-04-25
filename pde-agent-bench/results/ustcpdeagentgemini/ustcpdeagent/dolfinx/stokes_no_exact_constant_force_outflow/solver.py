import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    comm = MPI.COMM_WORLD
    
    mesh_resolution = 64
    ksp_type = "preonly"
    pc_type = "lu"
    
    msh = mesh.create_unit_square(comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    
    nu = 0.4
    f_const = fem.Constant(msh, PETSc.ScalarType((1.0, 0.0)))
    
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    a = (nu*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
         - p*ufl.div(v)*ufl.dx
         - q*ufl.div(u)*ufl.dx)
    L = ufl.inner(f_const, v)*ufl.dx
    
    fdim = msh.topology.dim - 1
    
    def boundary_marker(x):
        return np.logical_or(np.isclose(x[0], 0.0), np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], 1.0)))
    
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, boundary_marker)
    u0 = fem.Function(V)
    u0.x.array[:] = 0.0
    bc_u = fem.dirichletbc(u0, fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets), W.sub(0))
    
    bcs = [bc_u]
    
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                            petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = bb_tree(msh, msh.topology.dim)
    cell_candidates = compute_collisions_points(tree, pts)
    colliding = compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells = []
    eval_map = []
    for i, pt in enumerate(pts):
        if len(colliding.links(i)) > 0:
            points_on_proc.append(pt)
            cells.append(colliding.links(i)[0])
            eval_map.append(i)
            
    u_vals = np.full((pts.shape[0], gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals[eval_map] = vals

    mag = np.linalg.norm(u_vals, axis=1)
    mag_grid = mag.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": 2,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": 1e-8,
        "iterations": 1,
    }
    
    return {
        "u": mag_grid,
        "solver_info": solver_info
    }
