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

    comm = MPI.COMM_WORLD
    N = 128
    degree = 2

    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(domain)
    pi = np.pi
    u_exact = ufl.sin(pi*x[0])*ufl.sin(pi*x[1]) + 0.2*ufl.sin(5*pi*x[0])*ufl.sin(4*pi*x[1])
    # f = -div(grad(u_exact))
    f_expr = (2*pi*pi)*ufl.sin(pi*x[0])*ufl.sin(pi*x[1]) + 0.2*(25+16)*pi*pi*ufl.sin(5*pi*x[0])*ufl.sin(4*pi*x[1])

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = f_expr*v*ufl.dx

    # Boundary condition (zero since u_exact is zero on unit square boundary)
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    rtol = 1e-10
    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": rtol},
        petsc_options_prefix="poisson_"
    )
    u_sol = problem.solve()
    iters = problem.solver.getIterationNumber()

    # Sample on output grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)])

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(domain, cand, pts)

    points_on_proc = []
    cells_on_proc = []
    idx_map = []
    for i in range(pts.shape[0]):
        links = coll.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            idx_map.append(i)

    vals = np.zeros(nx_out*ny_out)
    if len(points_on_proc) > 0:
        res = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        vals[idx_map] = res.flatten()

    u_grid = vals.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": int(iters),
        }
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 128, "ny": 128, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    out = solve(case_spec)
    t1 = time.time()
    u_grid = out["u"]
    xs = np.linspace(0, 1, 128)
    ys = np.linspace(0, 1, 128)
    XX, YY = np.meshgrid(xs, ys)
    u_exact = np.sin(np.pi*XX)*np.sin(np.pi*YY) + 0.2*np.sin(5*np.pi*XX)*np.sin(4*np.pi*YY)
    err = np.sqrt(np.mean((u_grid - u_exact)**2))
    linf = np.max(np.abs(u_grid - u_exact))
    print(f"Time: {t1-t0:.3f}s, RMSE: {err:.3e}, Linf: {linf:.3e}, iters: {out['solver_info']['iterations']}")
