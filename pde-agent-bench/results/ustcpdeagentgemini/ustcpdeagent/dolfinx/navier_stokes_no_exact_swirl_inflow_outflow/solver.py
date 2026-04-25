import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from basix.ufl import element as basix_element
from basix.ufl import mixed_element as basix_mixed_element
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Parameters
    nu = 0.22
    mesh_res = 64
    
    # Create mesh
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = domain.geometry.dim

    # Function spaces (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", domain.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", domain.topology.cell_name(), 1)
    W = fem.functionspace(domain, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    # Residual formulation
    x = ufl.SpatialCoordinate(domain)
    f = ufl.as_vector((0.0, 0.0))
    
    def eps(u):
        return ufl.sym(ufl.grad(u))
    def sigma(u, p):
        return 2.0 * nu * eps(u) - p * ufl.Identity(gdim)

    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )

    J = ufl.derivative(F, w)

    # Boundary Conditions
    bcs = []
    fdim = domain.topology.dim - 1
    
    # x0: u = [sin(pi*y), 0.2*sin(2*pi*y)]
    facets_x0 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs_x0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_x0)
    u_x0 = fem.Function(V)
    u_x0.interpolate(lambda x: np.vstack((np.sin(np.pi * x[1]), 0.2 * np.sin(2.0 * np.pi * x[1]))))
    bcs.append(fem.dirichletbc(u_x0, dofs_x0, W.sub(0)))
    
    # y0: u = [0, 0]
    facets_y0 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
    dofs_y0 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y0)
    u_y0 = fem.Function(V)
    u_y0.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_y0, dofs_y0, W.sub(0)))
    
    # y1: u = [0, 0]
    facets_y1 = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 1.0))
    dofs_y1 = fem.locate_dofs_topological((W.sub(0), V), fdim, facets_y1)
    u_y1 = fem.Function(V)
    u_y1.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(u_y1, dofs_y1, W.sub(0)))
    
    # Initialize with Stokes guess
    F_stokes = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    problem_stokes = petsc.NonlinearProblem(F_stokes, w, bcs=bcs, J=J,
                                      petsc_options_prefix="stokes_",
                                      petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    problem_stokes.solve()

    # Nonlinear solve
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 20,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps"
    }

    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                      petsc_options_prefix="ns_",
                                      petsc_options=petsc_options)
    
    # We could capture snes convergence but in NonlinearProblem.solve() it returns the number of iterations
    iters, converged = problem.solve()
    w.x.scatter_forward()

    # Interpolate onto output grid
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
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_sol = w.sub(0).collapse()
    
    u_out = np.zeros(nx_out * ny_out)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        mags = np.linalg.norm(vals, axis=1)
        u_out[eval_map] = mags

    # Collect globally if in parallel, but usually benchmark runs in serial
    # Assuming serial execution:
    u_grid = u_out.reshape((ny_out, nx_out))

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": iters,
        "nonlinear_iterations": [iters]
    }

    return {
        "u": u_grid,
        "solver_info": solver_info
    }
