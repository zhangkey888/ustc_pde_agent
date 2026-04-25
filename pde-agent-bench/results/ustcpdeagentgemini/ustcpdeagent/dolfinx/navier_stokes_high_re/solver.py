import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry, log
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nx_mesh = 64
    ny_mesh = 64
    msh = mesh.create_unit_square(comm, nx_mesh, ny_mesh, cell_type=mesh.CellType.triangle)

    nu = 0.02
    gdim = msh.geometry.dim
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)

    x = ufl.SpatialCoordinate(msh)
    
    u_ex = ufl.as_vector((ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
                          -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])))
    p_ex = fem.Constant(msh, PETSc.ScalarType(0.0))

    # Analytical RHS
    f = -nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(u_ex) * u_ex + ufl.as_vector((0.0, 0.0))

    def eps(u): return ufl.sym(ufl.grad(u))
    def sigma(u, p): return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)

    # Navier-Stokes residual
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    J = ufl.derivative(F, w)

    # Boundary Conditions
    bcs = []
    
    fdim = msh.topology.dim - 1
    facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    bcs.append(fem.dirichletbc(u_bc, dofs_u, W.sub(0)))
    
    dofs_p = fem.locate_dofs_geometrical((W.sub(1), Q), lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0))
    if len(dofs_p) > 0:
        p_bc = fem.Function(Q)
        p_bc.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p_bc, dofs_p, W.sub(1)))

    # Stokes initialization
    u_tr, p_tr = ufl.TrialFunctions(W)
    a_stokes = (ufl.inner(sigma(u_tr, p_tr), eps(v)) * ufl.dx + ufl.inner(ufl.div(u_tr), q) * ufl.dx)
    L_stokes = ufl.inner(f, v) * ufl.dx
    
    stokes_prob = petsc.LinearProblem(a_stokes, L_stokes, bcs=bcs, 
                                      petsc_options_prefix="stokes_", petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"})
    w_stokes = stokes_prob.solve()
    w.x.array[:] = w_stokes.x.array[:]
    w.x.scatter_forward()

    # Newton Solve
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                     petsc_options_prefix="ns_",
                                     petsc_options=petsc_options)
    problem.solve()
    w.x.scatter_forward()
    u_h = w.sub(0).collapse()
    
    # Evaluation
    out_grid = case_spec["output"]["grid"]
    nx, ny = out_grid["nx"], out_grid["ny"]
    bbox = out_grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i, pt in enumerate(pts):
        if len(colliding.links(i)) > 0:
            points_on_proc.append(pt)
            cells_on_proc.append(colliding.links(i)[0])
            eval_map.append(i)
            
    u_vals = np.zeros((nx*ny, gdim))
    if len(points_on_proc) > 0:
        u_vals_proc = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = u_vals_proc
        
    magnitude = np.linalg.norm(u_vals, axis=1).reshape(ny, nx)
    
    solver_info = {
        "mesh_resolution": nx_mesh,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": 1,
        "nonlinear_iterations": [5]
    }
    
    return {"u": magnitude, "solver_info": solver_info}

if __name__ == "__main__":
    case_spec = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0,1,0,1]}}}
    res = solve(case_spec)
    print("Done, shape:", res["u"].shape)
