import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # Grid info
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]

    # Time parameters
    t0 = 0.0
    t_end = 0.1
    dt_val = 0.005  # smaller than suggested for better accuracy
    n_steps = int(round((t_end - t0) / dt_val))

    # Mesh
    N = 96
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)

    # Function space - P2 for accuracy
    degree = 2
    V = fem.functionspace(domain, ("Lagrange", degree))

    kappa = 1.0
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))

    # Exact solution UFL
    def u_exact_ufl(tc):
        return ufl.exp(-tc) * (
            ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
            + 0.2 * ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(6 * ufl.pi * x[1])
        )

    # Source term: f = du/dt - kappa * laplacian(u)
    # u = exp(-t)*(A + B), A = sin(pi*x)sin(pi*y), B = 0.2*sin(6pi x)sin(6pi y)
    # du/dt = -u
    # laplacian(A) = -2*pi^2 * A_base (with A = exp(-t)*A_base)
    # laplacian(B_base) = -72*pi^2 * B_base_spatial
    # So f = -u - kappa*( -2*pi^2*exp(-t)*sin(pi x)sin(pi y) - 72*pi^2*exp(-t)*0.2*sin(6pi x)sin(6pi y) )
    #      = -u + kappa*exp(-t)*( 2*pi^2*sin(pi x)sin(pi y) + 14.4*pi^2*sin(6pi x)sin(6pi y) )
    def f_ufl(tc):
        u_e = u_exact_ufl(tc)
        A_base = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        B_base = ufl.sin(6 * ufl.pi * x[0]) * ufl.sin(6 * ufl.pi * x[1])
        lap_u = ufl.exp(-tc) * (-2 * ufl.pi**2 * A_base + 0.2 * (-72 * ufl.pi**2) * B_base)
        return -u_e - kappa * lap_u

    # Initial condition
    u_n = fem.Function(V)
    u_expr_init = fem.Expression(u_exact_ufl(fem.Constant(domain, PETSc.ScalarType(0.0))),
                                  V.element.interpolation_points)
    u_n.interpolate(u_expr_init)

    # Boundary condition function
    u_bc = fem.Function(V)
    u_bc_expr = fem.Expression(u_exact_ufl(t_const), V.element.interpolation_points)
    u_bc.interpolate(u_bc_expr)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(u_bc, boundary_dofs)

    # Variational form (backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt_val))
    kappa_c = fem.Constant(domain, PETSc.ScalarType(kappa))

    a = (u * v + dt_c * kappa_c * ufl.inner(ufl.grad(u), ufl.grad(v))) * ufl.dx
    L = (u_n + dt_c * f_ufl(t_const)) * v * ufl.dx

    u_sol = fem.Function(V)

    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    b = petsc.create_vector(L_form.function_spaces)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("cg")
    pc = solver.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=500)

    total_iters = 0
    t_cur = t0
    for step in range(n_steps):
        t_cur = t0 + (step + 1) * dt_val
        t_const.value = t_cur

        # Update BC
        u_bc.interpolate(u_bc_expr)

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

        u_n.x.array[:] = u_sol.x.array

    # Sample on uniform grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]

    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)

    points_on_proc = []
    cells = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells.append(links[0])
            eval_map.append(i)

    u_vals_full = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        u_vals_full[eval_map] = vals.flatten()

    u_grid = u_vals_full.reshape(ny_out, nx_out)

    # Initial condition grid
    u_init_vals = np.full((pts.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u_n.eval(np.array(points_on_proc), np.array(cells, dtype=np.int32))
        # u_n was overwritten; recompute initial from exact formula
    # Compute exact initial on grid (since u_n was overwritten)
    t_init = 0.0
    A0 = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    B0 = 0.2 * np.sin(6 * np.pi * XX) * np.sin(6 * np.pi * YY)
    u_initial_grid = np.exp(-t_init) * (A0 + B0)

    # Verification: compare with exact at t_end
    A_e = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    B_e = 0.2 * np.sin(6 * np.pi * XX) * np.sin(6 * np.pi * YY)
    u_exact_grid = np.exp(-t_end) * (A_e + B_e)
    err = np.sqrt(np.mean((u_grid - u_exact_grid) ** 2))
    print(f"RMS error vs exact at t_end: {err:.3e}")
    print(f"Total linear iterations: {total_iters}")

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
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
    case_spec = {
        "output": {
            "grid": {
                "nx": 64,
                "ny": 64,
                "bbox": [0.0, 1.0, 0.0, 1.0],
            }
        },
        "pde": {"time": True},
    }
    t0 = time.time()
    result = solve(case_spec)
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Shape: {result['u'].shape}")
