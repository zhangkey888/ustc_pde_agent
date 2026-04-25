import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    mesh_res = 64
    element_degree = 3
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-10

    domain = mesh.create_rectangle(
        comm,
        [np.array([xmin, ymin]), np.array([xmax, ymax])],
        [mesh_res, mesh_res],
        cell_type=mesh.CellType.triangle,
    )

    V = fem.functionspace(domain, ("Lagrange", element_degree))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(domain)
    u_exact_ufl = ufl.sin(ufl.pi * x[0] * x[1])
    f_ufl = ufl.pi**2 * (x[0]**2 + x[1]**2) * ufl.sin(ufl.pi * x[0] * x[1])

    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_ufl * v * ufl.dx

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)

    u_bc = fem.Function(V)
    u_bc.interpolate(
        fem.Expression(u_exact_ufl, V.element.interpolation_points)
    )
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    problem = petsc.LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "ksp_rtol": rtol,
            "ksp_atol": 1e-14,
        },
        petsc_options_prefix="poisson_",
    )
    u_sol = problem.solve()

    ksp = problem.solver
    iterations = ksp.getIterationNumber()

    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

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

    u_values = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        pts_array = np.array(points_on_proc)
        cells_array = np.array(cells_on_proc, dtype=np.int32)
        vals = u_sol.eval(pts_array, cells_array)
        u_values[eval_map] = vals.flatten()

    if comm.size > 1:
        from mpi4py import MPI as MPI4
        recv_buf = np.full((pts.shape[0],), np.nan)
        comm.Allreduce(u_values, recv_buf, op=MPI4.MAX)
        u_values = recv_buf
        nan_mask = np.isnan(u_values)
        if np.any(nan_mask):
            recv_buf2 = np.full((pts.shape[0],), np.nan)
            comm.Allreduce(u_values, recv_buf2, op=MPI4.MIN)
            u_values = np.where(np.isnan(u_values), recv_buf2, u_values)

    u_grid = u_values.reshape(ny_out, nx_out)

    L2_error = fem.assemble_scalar(
        fem.form((u_sol - u_exact_ufl)**2 * ufl.dx)
    )
    L2_error = np.sqrt(comm.allreduce(L2_error, op=MPI.SUM))

    if comm.rank == 0:
        print(f"L2 error: {L2_error:.6e}")
        print(f"KSP iterations: {iterations}")

    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }

    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        }
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.3f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"solver_info: {result['solver_info']}")
