import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    eps_val = 0.01
    beta_val = np.array([-12.0, 6.0])

    # Mesh
    N = 208
    degree = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.exp(x[0]) * ufl.sin(ufl.pi * x[1])

    # f = -eps*laplacian(u) + beta . grad(u)
    grad_u_ex = ufl.as_vector([ufl.exp(x[0])*ufl.sin(ufl.pi*x[1]),
                                ufl.pi*ufl.exp(x[0])*ufl.cos(ufl.pi*x[1])])
    lap_u_ex = ufl.exp(x[0])*ufl.sin(ufl.pi*x[1]) - (ufl.pi**2)*ufl.exp(x[0])*ufl.sin(ufl.pi*x[1])

    beta = fem.Constant(msh, PETSc.ScalarType(beta_val))
    eps_c = fem.Constant(msh, PETSc.ScalarType(eps_val))

    f_expr = -eps_c * lap_u_ex + ufl.dot(beta, grad_u_ex)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Galerkin
    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(msh)
    bnorm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_h = bnorm * h / (2.0 * eps_c)
    tau = h / (2.0 * bnorm) * (1.0 / ufl.tanh(Pe_h) - 1.0 / Pe_h)

    r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f_expr
    supg_test = tau * ufl.dot(beta, ufl.grad(v))

    a_supg = ufl.inner(-eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)), supg_test) * ufl.dx
    L_supg = ufl.inner(f_expr, supg_test) * ufl.dx

    a_total = a + a_supg
    L_total = L + L_supg

    # Boundary condition from exact
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact, V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(msh, fdim,
        lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    ksp_type = "preonly"
    pc_type = "lu"

    problem = petsc.LinearProblem(
        a_total, L_total, bcs=[bc],
        petsc_options={"ksp_type": ksp_type, "pc_type": pc_type},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()

    its = problem.solver.getIterationNumber()

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]; ny = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    u_vals = np.zeros(nx*ny)
    if points_on_proc:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()
    u_grid = u_vals.reshape(ny, nx)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": 0.0,
            "iterations": int(its),
        }
    }
