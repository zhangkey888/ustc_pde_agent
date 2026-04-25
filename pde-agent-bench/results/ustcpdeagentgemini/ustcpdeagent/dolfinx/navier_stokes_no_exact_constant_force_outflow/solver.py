import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from basix.ufl import element as basix_element
from basix.ufl import mixed_element as basix_mixed_element
from dolfinx.fem import petsc
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    comm = MPI.COMM_WORLD
    
    mesh_res = 64
    domain = mesh.create_rectangle(comm, [[0.0, 0.0], [1.0, 1.0]], [mesh_res, mesh_res], cell_type=mesh.CellType.triangle)
    
    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    nu = 0.3
    f = fem.Constant(domain, PETSc.ScalarType((1.0, 0.0)))
    
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u, p):
        return 2.0 * nu * eps(u) - p * ufl.Identity(domain.geometry.dim)
    
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    
    def on_y0(x): return np.isclose(x[1], 0.0)
    def on_y1(x): return np.isclose(x[1], 1.0)
    def on_x1(x): return np.isclose(x[0], 1.0)
    def on_walls(x): return on_y0(x) | on_y1(x) | on_x1(x)
    
    wall_facets = mesh.locate_entities_boundary(domain, fdim, on_walls)
    dofs_wall = fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets)
    
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc_wall = fem.dirichletbc(u_bc, dofs_wall, W.sub(0))
    
    bcs = [bc_wall]
    
    # Pressure pinning: fix p(0.5, 0.5) to 0 (since it's an outflow problem, it has a natural BC at x=0 which fixes pressure. Let's not pin pressure if not fully enclosed. Actually, since x=0 has no Dirichlet, we don't strictly need pressure pinning. Wait, outflow BC is p*n - nu*du/dn = 0. This fixes the pressure scale! So NO pressure pinning is needed here.)
    
    w.x.array[:] = 0.0
    J = ufl.derivative(F, w)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
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
    
    snes = problem.solver
    snes.solve(None, w.x.petsc_vec)
    w.x.scatter_forward()
    
    iters = snes.getIterationNumber()
    linear_iters = snes.getLinearSolveIterations()
    
    u_h, p_h = w.sub(0).collapse(), w.sub(1).collapse()
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.zeros((pts.shape[1], 2))
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
        
    u_mag = np.linalg.norm(u_values, axis=1).reshape((ny_out, nx_out))
    
    return {
        "u": u_mag,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": linear_iters,
            "nonlinear_iterations": [iters],
        }
    }
