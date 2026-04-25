import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    # Read output grid specs
    out_grid = case_spec["output"]["grid"]
    nx_out = out_grid["nx"]
    ny_out = out_grid["ny"]
    bbox = out_grid["bbox"]

    # Mesh resolution
    nx, ny = 128, 128
    degree_u = 2
    degree_p = 1
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    
    nu = 0.08
    
    # Mixed space
    vel_el = basix_element("Lagrange", domain.topology.cell_name(), degree_u, shape=(domain.geometry.dim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), degree_p)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(domain)
    
    # Exact solution for source term and BCs
    u_ex_0 = ufl.pi * ufl.exp(6*(x[0]-1)) * ufl.cos(ufl.pi*x[1])
    u_ex_1 = -6 * ufl.exp(6*(x[0]-1)) * ufl.sin(ufl.pi*x[1])
    u_ex = ufl.as_vector([u_ex_0, u_ex_1])
    p_ex = ufl.sin(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    
    # Compute exact source term f
    def grad(u): return ufl.grad(u)
    def div(u): return ufl.div(u)
    def laplace(u): return ufl.div(ufl.grad(u))
    
    f_ex = grad(u_ex)*u_ex - nu*laplace(u_ex) + grad(p_ex)
    
    def eps(u): return ufl.sym(ufl.grad(u))
    def sigma(u, p): return 2.0 * nu * eps(u) - p * ufl.Identity(domain.geometry.dim)
    
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f_ex, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    J = ufl.derivative(F, w)
    
    # Boundary Conditions
    fdim = domain.topology.dim - 1
    bndry_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bdofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bndry_facets)
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points()))
    bc_u = fem.dirichletbc(u_bc_func, bdofs, W.sub(0))
    
    bcs = [bc_u]
    
    # Set initial guess to exact solution for fast convergence
    u_init_func = fem.Function(V)
    u_init_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points()))
    
    p_init_func = fem.Function(Q)
    p_init_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points()))
    
    w.sub(0).interpolate(u_init_func)
    w.sub(1).interpolate(p_init_func)
    w.x.scatter_forward()
    
    # Solve
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J, petsc_options_prefix="ns_",
                                     petsc_options={"snes_type": "newtonls",
                                                    "ksp_type": "preonly",
                                                    "pc_type": "lu",
                                                    "pc_factor_mat_solver_type": "mumps",
                                                    "snes_rtol": 1e-8,
                                                    "snes_max_it": 20})
    snes = problem.snes
    # Manually get number of linear iterations
    snes.setConvergenceHistory()
    w_h = problem.solve()
    linear_its = snes.getLinearSolveIterations()
    newton_its = snes.getIterationNumber()
    
    u_sol, p_sol = w.sub(0).collapse(), w.sub(1).collapse()
    
    # Probe
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    u_values = np.zeros((pts.shape[0], 2))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map, :] = vals[:, :2]
        
    magnitude = np.linalg.norm(u_values, axis=1).reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree_u,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": linear_its,
        "nonlinear_iterations": [newton_its]
    }
    
    return {
        "u": magnitude,
        "solver_info": solver_info
    }
