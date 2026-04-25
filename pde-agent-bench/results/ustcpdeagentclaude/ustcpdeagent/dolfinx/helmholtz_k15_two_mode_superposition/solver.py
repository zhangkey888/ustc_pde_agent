import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    k_val = 15.0

    # Mesh resolution - P2 elements, need ~10 dof per wavelength
    # wavelength ~ 2pi/15 ~ 0.42, need ~2.4 per unit length
    # With P2, N=128 gives 256 effective dofs per direction
    N = 160
    degree = 3

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    # u_exact = sin(2pi x)sin(pi y) + sin(pi x) sin(3pi y)
    u_exact = (ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
               + ufl.sin(ufl.pi*x[0]) * ufl.sin(3*ufl.pi*x[1]))

    # -laplacian(u) - k^2 u = f
    # laplacian of sin(a x) sin(b y) = -(a^2+b^2) sin(ax)sin(by)
    # -laplacian = (a^2+b^2) sin(ax)sin(by)
    # f = (a^2+b^2 - k^2) sin(ax) sin(by)
    k2 = k_val**2
    f_expr = ((4*ufl.pi**2 + ufl.pi**2 - k2) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
              + (ufl.pi**2 + 9*ufl.pi**2 - k2) * ufl.sin(ufl.pi*x[0]) * ufl.sin(3*ufl.pi*x[1]))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k_const = fem.Constant(msh, PETSc.ScalarType(k_val))

    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k_const**2 * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    # BC: u = u_exact on boundary (which is zero actually, since sin(0)=sin(pi)=0)
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, dofs)

    petsc_opts = {"ksp_type": "preonly", "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps"}
    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                   petsc_options=petsc_opts,
                                   petsc_options_prefix="helm_")
    u_sol = problem.solve()

    # Sample on output grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]; ny = grid["ny"]; bbox = grid["bbox"]
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
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.zeros(nx*ny)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny, nx)

    # Accuracy check
    u_exact_vals = (np.sin(2*np.pi*XX)*np.sin(np.pi*YY)
                    + np.sin(np.pi*XX)*np.sin(3*np.pi*YY))
    err = np.sqrt(np.mean((u_grid - u_exact_vals)**2))
    print(f"RMS error on grid: {err:.3e}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": 1,
        }
    }


if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 128, "ny": 128, "bbox": [0,1,0,1]}}}
    t0 = time.time()
    out = solve(spec)
    print(f"Wall time: {time.time()-t0:.2f}s")
    print(f"Shape: {out['u'].shape}")
