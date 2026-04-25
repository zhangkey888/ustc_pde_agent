import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    eps_val = 0.05
    beta_vec = np.array([4.0, 2.0])

    comm = MPI.COMM_WORLD
    N = 64
    degree = 2
    p0 = np.array([0.0, 0.0])
    p1 = np.array([1.0, 1.0])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N],
                                    cell_type=mesh.CellType.quadrilateral)

    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi
    u_exact = ufl.sin(2*pi*x[0]) * ufl.sin(pi*x[1])
    grad_u = ufl.as_vector([2*pi*ufl.cos(2*pi*x[0])*ufl.sin(pi*x[1]),
                             pi*ufl.sin(2*pi*x[0])*ufl.cos(pi*x[1])])
    lap_u = -(4*pi*pi + pi*pi) * u_exact
    beta = fem.Constant(domain, PETSc.ScalarType(beta_vec))
    eps_c = fem.Constant(domain, PETSc.ScalarType(eps_val))
    f_expr = -eps_c * lap_u + ufl.dot(beta, grad_u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = eps_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
        + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx
    L = f_expr * v * ufl.dx

    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    beta_norm = ufl.sqrt(ufl.dot(beta, beta))
    Pe_local = beta_norm * h / (2 * eps_c)
    # tau according to standard formula
    tau = (h / (2 * beta_norm)) * (1.0 / ufl.tanh(Pe_local) - 1.0/Pe_local)

    # Strong residual (for degree>=2, laplacian term included)
    if degree >= 2:
        r_u = -eps_c * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u))
        r_f = f_expr
    else:
        r_u = ufl.dot(beta, ufl.grad(u))
        r_f = f_expr

    a += tau * ufl.inner(r_u, ufl.dot(beta, ufl.grad(v))) * ufl.dx
    L += tau * ufl.inner(r_f, ufl.dot(beta, ufl.grad(v))) * ufl.dx

    # BC from exact solution
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "gmres", "pc_type": "hypre", "ksp_rtol": 1e-10},
        petsc_options_prefix="cd_"
    )
    u_sol = problem.solve()
    its = problem.solver.getIterationNumber()

    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cc = geometry.compute_collisions_points(tree, pts)
    col = geometry.compute_colliding_cells(domain, cc, pts)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = col.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    u_values = np.zeros(pts.shape[0])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc),
                          np.array(cells_on_proc, dtype=np.int32))
        u_values[idx_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(its),
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}}
    t0 = time.time()
    res = solve(spec)
    t1 = time.time()
    print(f"Wall: {t1-t0:.3f}s")
    # check error
    xs = np.linspace(0, 1, 64)
    ys = np.linspace(0, 1, 64)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.sin(2*np.pi*XX) * np.sin(np.pi*YY)
    err = np.sqrt(np.mean((res["u"] - u_ex)**2))
    print(f"L2 err: {err:.3e}")
    print(f"Max err: {np.max(np.abs(res['u']-u_ex)):.3e}")
    print(res["solver_info"])
