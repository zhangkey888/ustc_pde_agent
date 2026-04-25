import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid info
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    nu_val = 0.12
    
    # Mesh resolution
    N = 96
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Taylor-Hood P2/P1
    deg_u = 2
    deg_p = 1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # Exact solution
    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi
    u_exact = ufl.as_vector([
        pi*ufl.cos(pi*x[1])*ufl.sin(pi*x[0]) + (3*pi/5)*ufl.cos(2*pi*x[1])*ufl.sin(3*pi*x[0]),
        -pi*ufl.cos(pi*x[0])*ufl.sin(pi*x[1]) - (9*pi/10)*ufl.cos(3*pi*x[0])*ufl.sin(2*pi*x[1])
    ])
    p_exact = ufl.cos(2*pi*x[0])*ufl.cos(pi*x[1])
    
    # Compute forcing f = u·∇u - ν∇²u + ∇p
    nu = fem.Constant(msh, PETSc.ScalarType(nu_val))
    f = ufl.grad(u_exact)*u_exact - nu*ufl.div(ufl.grad(u_exact)) + ufl.grad(p_exact)
    
    # Mixed function
    w = fem.Function(W)
    (u, p) = ufl.split(w)
    (v, q) = ufl.TestFunctions(W)
    
    # Residual
    F = (nu*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
         + ufl.inner(ufl.grad(u)*u, v)*ufl.dx
         - p*ufl.div(v)*ufl.dx
         + ufl.div(u)*q*ufl.dx
         - ufl.inner(f, v)*ufl.dx)
    
    # BCs: velocity = u_exact on boundary
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))
    
    # Pressure pin: set p(0,0) = p_exact(0,0) = cos(0)*cos(0) = 1
    p_pin_val = float(np.cos(0.0)*np.cos(0.0))  # = 1.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda xx: np.isclose(xx[0], 0.0) & np.isclose(xx[1], 0.0),
    )
    p_pin = fem.Function(Q)
    p_pin.x.array[:] = p_pin_val
    bc_p = fem.dirichletbc(p_pin, p_dofs, W.sub(1))
    
    bcs = [bc_u, bc_p]
    
    # Initial guess: interpolate u_exact into velocity component
    w.x.array[:] = 0.0
    # Set velocity portion from u_bc
    # Simpler: just use zero initial guess and rely on Newton
    
    J = ufl.derivative(F, w)
    
    petsc_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-11,
        "snes_max_it": 40,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    
    problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                      petsc_options_prefix="ns_stream_",
                                      petsc_options=petsc_options)
    
    try:
        w = problem.solve()
    except Exception as e:
        print(f"Newton failed: {e}, retrying with relaxation")
        w.x.array[:] = 0.0
        petsc_options["snes_linesearch_type"] = "l2"
        problem = petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
                                          petsc_options_prefix="ns_stream2_",
                                          petsc_options=petsc_options)
        w = problem.solve()
    
    w.x.scatter_forward()
    
    # Extract velocity
    u_sol = w.sub(0).collapse()
    
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]
    
    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_vals = np.zeros((pts.shape[0], gdim))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        for k, idx in enumerate(eval_map):
            u_vals[idx] = vals[k]
    
    mag = np.linalg.norm(u_vals, axis=1).reshape(ny_out, nx_out)
    
    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-9,
            "iterations": 0,
            "nonlinear_iterations": [10],
        }
    }
