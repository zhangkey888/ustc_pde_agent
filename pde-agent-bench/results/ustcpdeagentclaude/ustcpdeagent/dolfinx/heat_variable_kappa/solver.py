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
    dt_val = 0.005  # refined from suggested 0.01
    n_steps = int(round((t_end - t0) / dt_val))

    nx_mesh = 64
    degree = 2

    msh = mesh.create_unit_square(comm, nx_mesh, nx_mesh, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(msh, ("Lagrange", degree))

    x = ufl.SpatialCoordinate(msh)
    t_const = fem.Constant(msh, PETSc.ScalarType(t0))

    # Exact solution: u = exp(-t)*sin(2*pi*x)*sin(2*pi*y)
    u_exact_expr = ufl.exp(-t_const) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])

    # kappa = 1 + 0.3*cos(2*pi*x)*cos(2*pi*y)
    kappa = 1.0 + 0.3 * ufl.cos(2*ufl.pi*x[0]) * ufl.cos(2*ufl.pi*x[1])

    # Compute f = du/dt - div(kappa * grad(u))
    # du/dt = -exp(-t)*sin(2pi x)*sin(2pi y)
    dudt = -u_exact_expr
    # div(kappa * grad(u))
    div_kgrad = ufl.div(kappa * ufl.grad(u_exact_expr))
    f_expr = dudt - div_kgrad

    # Initial condition
    u_n = fem.Function(V)
    u_init_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)
    t_const.value = t0
    u_n.interpolate(u_init_expr)

    # Boundary condition - interpolate exact solution
    u_bc = fem.Function(V)
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_expr = fem.Expression(u_exact_expr, V.element.interpolation_points)

    # Variational form - Backward Euler
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(msh, PETSc.ScalarType(dt_val))

    a = u*v*ufl.dx + dt_c*ufl.inner(kappa*ufl.grad(u), ufl.grad(v))*ufl.dx
    L = u_n*v*ufl.dx + dt_c*f_expr*v*ufl.dx

    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble matrix once
    t_const.value = t0 + dt_val  # matrix uses kappa which is time-independent; doesn't matter
    u_bc.interpolate(bc_expr)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)

    u_sol = fem.Function(V)

    # Save initial for output
    t_const.value = t0
    u_init_save = fem.Function(V)
    u_init_save.interpolate(u_init_expr)

    total_iters = 0
    t = t0
    for step in range(n_steps):
        t_new = t + dt_val
        t_const.value = t_new
        # update BC
        u_bc.interpolate(bc_expr)

        # Assemble RHS
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iters += solver.getIterationNumber()

        u_n.x.array[:] = u_sol.x.array[:]
        t = t_new

    # Accuracy check via L2 error
    t_const.value = t
    err_form = fem.form((u_sol - u_exact_expr)**2 * ufl.dx)
    err_L2 = np.sqrt(comm.allreduce(fem.assemble_scalar(err_form), op=MPI.SUM))
    print(f"L2 error at t={t}: {err_L2}")

    # Sample on output grid
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    cells = []
    points_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    u_grid = np.full(nx_out*ny_out, np.nan)
    u_init_grid = np.full(nx_out*ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_grid[eval_map] = vals.flatten()
        vals0 = u_init_save.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_init_grid[eval_map] = vals0.flatten()

    u_grid = u_grid.reshape(ny_out, nx_out)
    u_init_grid = u_init_grid.reshape(ny_out, nx_out)

    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": nx_mesh,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": int(total_iters),
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        },
    }


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    t0 = time.time()
    result = solve(case_spec)
    print(f"Wall time: {time.time()-t0:.2f}s")
    print(f"u shape: {result['u'].shape}")
    # Compare with exact at t=0.1
    nx, ny = 64, 64
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys)
    u_ex = np.exp(-0.1) * np.sin(2*np.pi*XX) * np.sin(2*np.pi*YY)
    err = np.sqrt(np.mean((result["u"] - u_ex)**2))
    print(f"Grid RMS error: {err}")
    print(f"Max error: {np.max(np.abs(result['u']-u_ex))}")
