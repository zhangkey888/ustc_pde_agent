import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Parameters
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.001  # smaller dt for accuracy
    kappa = 1.0

    # Mesh & space
    N = 128
    degree = 2
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(t0))
    # Source: f = exp(-t)*sin(4pi x)*sin(4pi y)*(32pi^2 - 1)
    f_expr = ufl.exp(-t_const) * ufl.sin(4*ufl.pi*x[0]) * ufl.sin(4*ufl.pi*x[1]) * (32.0*ufl.pi**2 - 1.0)

    # Initial condition
    u_n = fem.Function(V)
    u_exact_ufl_t0 = ufl.sin(4*ufl.pi*x[0]) * ufl.sin(4*ufl.pi*x[1])
    u_n.interpolate(fem.Expression(u_exact_ufl_t0, V.element.interpolation_points))

    # BC: zero everywhere on boundary (since sin(4*pi*x)=0 at x=0,1)
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bc = fem.dirichletbc(u_bc, fem.locate_dofs_topological(V, fdim, boundary_facets))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_const = fem.Constant(msh, PETSc.ScalarType(dt_val))

    # Backward Euler: (u - u_n)/dt - kappa*div(grad(u)) = f
    a = (u * v + dt_const * kappa * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n + dt_const * f_expr) * v * ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-14, max_it=1000)

    u_sol = fem.Function(V)

    t = t0
    n_steps = int(round((t_end - t0) / dt_val))
    total_iters = 0

    for step in range(n_steps):
        t += dt_val
        t_const.value = t

        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array

    # Sample on grid
    grid = case_spec["output"]["grid"]
    nx = grid["nx"]
    ny = grid["ny"]
    bbox = grid["bbox"]
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

    u_vals = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()
    u_grid = u_vals.reshape(ny, nx)

    # Initial condition grid
    u_init_vals = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        # evaluate initial condition analytically
        pass
    # analytical initial
    u_init = np.sin(4*np.pi*XX) * np.sin(4*np.pi*YY)

    # Error check against analytical
    u_exact_grid = np.exp(-t_end) * np.sin(4*np.pi*XX) * np.sin(4*np.pi*YY)
    err = np.linalg.norm(u_grid - u_exact_grid) / np.sqrt(nx*ny)
    print(f"L2 grid error: {err:.4e}, t={t}, steps={n_steps}")

    return {
        "u": u_grid,
        "u_initial": u_init,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iters,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0, 1, 0, 1]}}
    }
    t0 = time.time()
    result = solve(spec)
    print(f"Wall time: {time.time()-t0:.2f}s")
    print("u shape:", result["u"].shape)
