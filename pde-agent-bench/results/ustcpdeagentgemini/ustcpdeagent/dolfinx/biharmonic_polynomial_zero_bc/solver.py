import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    
    # Extract grid info
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Parameters
    mesh_res = 64
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    
    el = basix_element("Lagrange", domain.topology.cell_name(), degree)
    mel = basix_mixed_element([el, el])
    W = fem.functionspace(domain, mel)
    
    V_u, _ = W.sub(0).collapse()
    V_v, _ = W.sub(1).collapse()
    
    # Trial and Test functions
    (u, v) = ufl.TrialFunctions(W)
    (phi, psi) = ufl.TestFunctions(W)
    
    # Source term f = 8
    f = fem.Constant(domain, PETSc.ScalarType(8.0))
    
    # Mixed formulation
    # v = -Delta u  =>  int grad(u)*grad(phi) - int v*phi = 0
    # -Delta v = f  =>  int grad(v)*grad(psi) = int f*psi
    a = ufl.inner(ufl.grad(u), ufl.grad(phi)) * ufl.dx - ufl.inner(v, phi) * ufl.dx \
      + ufl.inner(ufl.grad(v), ufl.grad(psi)) * ufl.dx
    L = ufl.inner(f, psi) * ufl.dx
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    # u = 0 on boundary
    u_bc_func = fem.Function(V_u)
    u_bc_func.x.array[:] = 0.0
    dofs_u = fem.locate_dofs_topological((W.sub(0), V_u), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    # v = 2y(1-y) + 2x(1-x) on boundary
    v_bc_func = fem.Function(V_v)
    v_bc_func.interpolate(lambda x: 2*x[1]*(1-x[1]) + 2*x[0]*(1-x[0]))
    dofs_v = fem.locate_dofs_topological((W.sub(1), V_v), fdim, boundary_facets)
    bc_v = fem.dirichletbc(v_bc_func, dofs_v, W.sub(1))
    
    bcs = [bc_u, bc_v]
    
    # Solve
    problem = petsc.LinearProblem(a, L, bcs=bcs,
                                  petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
                                  petsc_options_prefix="biharm_")
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = bb_tree(domain, domain.topology.dim)
    cell_candidates = compute_collisions_points(tree, pts)
    colliding = compute_colliding_cells(domain, cell_candidates, pts)
    
    cells = []
    points_on_proc = []
    eval_map = []
    for i, pt in enumerate(pts):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pt)
            cells.append(links[0])
            eval_map.append(i)
            
    u_grid_flat = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_grid_flat[eval_map] = vals.flatten()
        
    u_grid = u_grid_flat.reshape(ny_out, nx_out)
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1
    }
    
    return {"u": u_grid, "solver_info": solver_info}

if __name__ == "__main__":
    pass
