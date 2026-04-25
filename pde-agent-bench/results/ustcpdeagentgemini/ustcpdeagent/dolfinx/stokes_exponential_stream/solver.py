import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    # Mesh resolution (higher to ensure accuracy constraint <= 1.68e-06 is met)
    N = 128
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim

    # Taylor-Hood elements
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, V_to_W = W.sub(0).collapse()
    Q, Q_to_W = W.sub(1).collapse()

    nu = 1.0

    # Exact solution for boundary conditions and source term
    x = ufl.SpatialCoordinate(msh)
    # Manufactured solution given:
    # u = ['pi*exp(x)*cos(pi*y)', '-exp(x)*sin(pi*y)']
    # p = exp(x)*cos(pi*y)
    u_exact = ufl.as_vector([
        ufl.pi * ufl.exp(x[0]) * ufl.cos(ufl.pi * x[1]),
        -ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    p_exact = ufl.exp(x[0]) * ufl.cos(ufl.pi * x[1])

    # Compute source term f = -nu * div(grad(u)) + grad(p)
    # -nu * div(grad(u)):
    # div(grad(u[0])) = div([pi*exp(x)*cos(pi*y), -pi^2*exp(x)*sin(pi*y)]) 
    #                 = pi*exp(x)*cos(pi*y) - pi^3*exp(x)*cos(pi*y) = pi*exp(x)*cos(pi*y)*(1 - pi^2)
    # div(grad(u[1])) = div([-exp(x)*sin(pi*y), -pi*exp(x)*cos(pi*y)]) 
    #                 = -exp(x)*sin(pi*y) + pi^2*exp(x)*sin(pi*y) = -exp(x)*sin(pi*y)*(1 - pi^2)
    # grad(p) = [exp(x)*cos(pi*y), -pi*exp(x)*sin(pi*y)]
    
    # f[0] = -1.0 * pi*exp(x)*cos(pi*y)*(1 - pi^2) + exp(x)*cos(pi*y) 
    #      = exp(x)*cos(pi*y) * (-pi + pi^3 + 1)
    
    # f[1] = -1.0 * (-exp(x)*sin(pi*y)*(1 - pi^2)) - pi*exp(x)*sin(pi*y)
    #      = exp(x)*sin(pi*y) * (1 - pi^2 - pi)

    f1 = ufl.exp(x[0]) * ufl.cos(ufl.pi * x[1]) * (1 - ufl.pi + ufl.pi**3)
    f2 = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1]) * (1 - ufl.pi**2 - ufl.pi)
    f = ufl.as_vector([f1, f2])

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # Velocity BC
    fdim = msh.topology.dim - 1
    wall_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    
    u0 = fem.Function(V)
    u_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u0.interpolate(u_expr)
    bc_u = fem.dirichletbc(u0, fem.locate_dofs_topological((W.sub(0), V), fdim, wall_facets), W.sub(0))

    # Pressure Pinning at origin
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    p0 = fem.Function(Q)
    # p_exact at (0,0) is exp(0)*cos(0) = 1.0
    p0.x.array[:] = 1.0
    
    # It is important that the pressure pin is applied to exactly one node
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p0, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Solve
    problem = LinearProblem(a, L, bcs=bcs,
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                            petsc_options_prefix="stokes_")
    w_h = problem.solve()
    u_h = w_h.sub(0).collapse()

    # Interpolate output
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
    tree = bb_tree(msh, msh.topology.dim)
    cell_candidates = compute_collisions_points(tree, pts)
    colliding = compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(len(pts)):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    u_vals = u_h.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
    magnitude = np.zeros(len(pts))
    magnitude[eval_map] = np.linalg.norm(u_vals, axis=1)
    magnitude_grid = magnitude.reshape((ny_out, nx_out))

    return {
        "u": magnitude_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-8,
            "iterations": 1
        }
    }

