import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time
import json
import argparse
import sys
from pathlib import Path

ScalarType = PETSc.ScalarType


def solve(case_spec: dict = None) -> dict:
    """
    Solve the transient heat equation using backward Euler time stepping.
    
    ∂u/∂t - ∇·(κ ∇u) = f   in Ω × (0, T]
      u = 0                   on ∂Ω
      u(x,0) = sin(2πx)sin(πy)
    
    f = cos(2πx)sin(πy), κ = 0.8
    """
    # ---- Default parameters ----
    kappa_val = 0.8
    t_end = 0.2
    dt_val = 0.02
    scheme = "backward_euler"
    nx_out = 50
    ny_out = 50

    # ---- Extract parameters from case_spec if provided ----
    if case_spec is not None:
        pde = case_spec.get("pde", {})
        coeffs = pde.get("coefficients", {})
        time_params = pde.get("time", {})

        # Diffusivity
        kappa_raw = coeffs.get("kappa", coeffs.get("κ", None))
        if kappa_raw is not None:
            if isinstance(kappa_raw, dict):
                kappa_val = float(kappa_raw.get("value", kappa_val))
            else:
                kappa_val = float(kappa_raw)

        # Time parameters
        t_end = float(time_params.get("t_end", t_end))
        dt_val = float(time_params.get("dt", dt_val))
        scheme = time_params.get("scheme", scheme)

        # Output grid
        output_spec = case_spec.get("output", {})
        grid_spec = output_spec.get("grid", {})
        nx_out = grid_spec.get("nx", nx_out)
        ny_out = grid_spec.get("ny", ny_out)

    # ---- Mesh and FE parameters ----
    N = 64
    element_degree = 1

    # ---- Create mesh ----
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1

    # ---- Function space ----
    V = fem.functionspace(domain, ("Lagrange", element_degree))

    # ---- Define functions ----
    u_n = fem.Function(V, name="u_n")  # solution at previous time step
    u_h = fem.Function(V, name="u_h")  # solution at current time step

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # ---- Spatial coordinate ----
    x = ufl.SpatialCoordinate(domain)

    # ---- Source term: f = cos(2*pi*x)*sin(pi*y) ----
    f_expr = ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # ---- Initial condition: u0 = sin(2*pi*x)*sin(pi*y) ----
    u0_expr = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])

    # Interpolate initial condition
    u0_fem_expr = fem.Expression(u0_expr, V.element.interpolation_points)
    u_n.interpolate(u0_fem_expr)

    # Store initial condition for output
    u_initial_func = fem.Function(V)
    u_initial_func.interpolate(u0_fem_expr)

    # ---- Boundary conditions: u = 0 on all boundaries ----
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]

    # ---- Time stepping parameters ----
    dt = fem.Constant(domain, ScalarType(dt_val))
    kappa_const = fem.Constant(domain, ScalarType(kappa_val))

    # ---- Backward Euler weak form ----
    # (u - u_n)/dt - kappa * laplacian(u) = f
    # Weak form: (u/dt)*v + kappa*grad(u).grad(v) = f*v + (u_n/dt)*v
    a = (u / dt) * v * ufl.dx + kappa_const * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f_expr * v * ufl.dx + (u_n / dt) * v * ufl.dx

    # ---- Compile forms ----
    a_form = fem.form(a)
    L_form = fem.form(L)

    # ---- Assemble matrix (constant in time for this problem) ----
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    # ---- Create RHS vector ----
    b_vec = petsc.create_vector(V)

    # ---- Setup KSP solver ----
    ksp_type_str = "cg"
    pc_type_str = "hypre"
    rtol_val = 1e-8

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type_str)
    pc = solver.getPC()
    pc.setType(pc_type_str)
    solver.setTolerances(rtol=rtol_val, atol=1e-12, max_it=1000)
    solver.setUp()

    # ---- Time stepping loop ----
    n_steps = int(np.round(t_end / dt_val))
    total_iterations = 0
    t = 0.0

    for step in range(n_steps):
        t += dt_val

        # Assemble RHS
        with b_vec.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b_vec, L_form)
        petsc.apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b_vec, bcs)

        # Solve
        solver.solve(b_vec, u_h.x.petsc_vec)
        u_h.x.scatter_forward()

        total_iterations += solver.getIterationNumber()

        # Update previous solution
        u_n.x.array[:] = u_h.x.array[:]

    # ---- Evaluate solution on output grid ----
    xs = np.linspace(0.0, 1.0, nx_out)
    ys = np.linspace(0.0, 1.0, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d

    # Point evaluation
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)

    u_grid_flat = np.full(points_3d.shape[0], np.nan)
    u_initial_flat = np.full(points_3d.shape[0], np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points_3d[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        u_grid_flat[eval_map] = vals.flatten()
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_initial_flat[eval_map] = vals_init.flatten()

    u_grid = u_grid_flat.reshape((nx_out, ny_out))
    u_initial_grid = u_initial_flat.reshape((nx_out, ny_out))

    # Cleanup
    solver.destroy()
    A.destroy()
    b_vec.destroy()

    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": ksp_type_str,
            "pc_type": pc_type_str,
            "rtol": rtol_val,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": scheme,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default=None)
    args, _ = parser.parse_known_args()

    t0 = time.time()
    result = solve()
    elapsed = time.time() - t0

    u = result["u"]
    info = result["solver_info"]

    if args.outdir is not None:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # Save solution.npz with x, y, u arrays
        nx_out, ny_out = u.shape
        xs = np.linspace(0.0, 1.0, nx_out)
        ys = np.linspace(0.0, 1.0, ny_out)
        np.savez(outdir / "solution.npz", x=xs, y=ys, u=u)

        # Save meta.json
        meta = {
            "solver_info": info,
            "wall_time": elapsed,
        }
        if result.get("u_initial") is not None:
            meta["has_u_initial"] = True
        with open(outdir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"Solution saved to {outdir}")
    else:
        print(f"Solution shape: {u.shape}")
        print(f"Solution min: {np.nanmin(u):.8f}, max: {np.nanmax(u):.8f}")
        print(f"Solution mean: {np.nanmean(u):.8f}")
        print(f"NaN count: {np.isnan(u).sum()}")

    print(f"Wall time: {elapsed:.3f}s")
    print(f"Solver info: {info}")
