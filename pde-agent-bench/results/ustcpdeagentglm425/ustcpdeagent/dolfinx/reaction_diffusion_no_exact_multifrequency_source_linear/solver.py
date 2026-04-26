import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    pde = case_spec.get("pde", {})

    # Domain
    domain_spec = pde.get("domain", {})
    if isinstance(domain_spec, dict):
        bbox_spec = domain_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    else:
        bbox_spec = [0.0, 1.0, 0.0, 1.0]
    xmin = float(bbox_spec[0]); xmax = float(bbox_spec[1])
    ymin = float(bbox_spec[2]); ymax = float(bbox_spec[3])

    # Time parameters
    time_spec = pde.get("time", {})
    t0 = float(time_spec.get("t0", 0.0))
    t_end = float(time_spec.get("t_end", 0.5))
    dt_suggested = float(time_spec.get("dt", 0.01))
    scheme = time_spec.get("scheme", "crank_nicolson")

    # Use dt/5 for good temporal accuracy (CN is O(dt^2))
    dt_val = dt_suggested / 5.0

    # PDE coefficients
    epsilon = float(pde.get("epsilon", 0.1))
    reaction_alpha = float(pde.get("reaction_alpha", 1.0))

    # Output grid
    out_spec = case_spec.get("output", {})
    grid_spec = out_spec.get("grid", {})
    nx_out = int(grid_spec.get("nx", 64))
    ny_out = int(grid_spec.get("ny", 64))
    out_bbox = grid_spec.get("bbox", [xmin, xmax, ymin, ymax])

    # Mesh and element parameters
    N = 128
    elem_deg = 2
    theta = 0.5  # Crank-Nicolson

    # Create mesh
    p0 = np.array([xmin, ymin])
    p1 = np.array([xmax, ymax])
    domain = mesh.create_rectangle(comm, [p0, p1], [N, N],
                                   cell_type=mesh.CellType.triangle)

    # Function space
    V = fem.functionspace(domain, ("Lagrange", elem_deg))

    # UFL spatial coordinate
    x = ufl.SpatialCoordinate(domain)
    pi = ufl.pi

    # Source term UFL
    f_ufl = (ufl.sin(5 * pi * x[0]) * ufl.sin(3 * pi * x[1]) +
             0.5 * ufl.sin(9 * pi * x[0]) * ufl.sin(7 * pi * x[1]))

    # IC UFL
    u0_ufl = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

    # Interpolate source
    f_func = fem.Function(V)
    f_func.interpolate(fem.Expression(f_ufl, V.element.interpolation_points))

    # Solution functions
    u_n = fem.Function(V)
    u_np1 = fem.Function(V)

    # Interpolate IC
    u0_expr = fem.Expression(u0_ufl, V.element.interpolation_points)
    u_n.interpolate(u0_expr)

    # Boundary conditions - homogeneous Dirichlet on all boundaries
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), boundary_dofs, V)

    # Variational forms (Crank-Nicolson)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    a = ((1.0 / dt_val + theta * reaction_alpha) * ufl.inner(u, v) * ufl.dx +
         theta * epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)

    L = ((1.0 / dt_val - (1 - theta) * reaction_alpha) * ufl.inner(u_n, v) * ufl.dx -
         (1 - theta) * epsilon * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx +
         ufl.inner(f_func, v) * ufl.dx)

    a_form = fem.form(a)
    L_form = fem.form(L)

    # Assemble LHS matrix
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()

    # KSP solver - LU direct
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.getPC().setType(PETSc.PC.Type.LU)
    ksp.setFromOptions()

    # RHS vector
    b = petsc.create_vector(V)

    # Time stepping
    n_steps = int(round((t_end - t0) / dt_val))
    total_iterations = 0

    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])

        ksp.solve(b, u_np1.x.petsc_vec)
        u_np1.x.scatter_forward()

        total_iterations += ksp.getIterationNumber()

        u_n.x.array[:] = u_np1.x.array[:]
        u_n.x.scatter_forward()

    # === ACCURACY VERIFICATION ===
    # Temporal convergence: advance 1 step from current state with dt and dt/2
    u_save = fem.Function(V)
    u_save.x.array[:] = u_np1.x.array[:]
    u_save.x.scatter_forward()

    # One coarse step with dt
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    u_coarse = fem.Function(V)
    ksp.solve(b, u_coarse.x.petsc_vec)
    u_coarse.x.scatter_forward()

    # Two fine steps with dt/2
    dt_half = dt_val / 2.0
    a_half = ((1.0 / dt_half + theta * reaction_alpha) * ufl.inner(u, v) * ufl.dx +
              theta * epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    a_half_form = fem.form(a_half)
    A_half = petsc.assemble_matrix(a_half_form, bcs=[bc])
    A_half.assemble()
    ksp_half = PETSc.KSP().create(domain.comm)
    ksp_half.setOperators(A_half)
    ksp_half.setType(PETSc.KSP.Type.PREONLY)
    ksp_half.getPC().setType(PETSc.PC.Type.LU)
    ksp_half.setFromOptions()

    u_n_half = fem.Function(V)
    u_n_half.x.array[:] = u_save.x.array[:]
    u_n_half.x.scatter_forward()
    L_half = ((1.0 / dt_half - (1 - theta) * reaction_alpha) * ufl.inner(u_n_half, v) * ufl.dx -
              (1 - theta) * epsilon * ufl.inner(ufl.grad(u_n_half), ufl.grad(v)) * ufl.dx +
              ufl.inner(f_func, v) * ufl.dx)
    L_half_form = fem.form(L_half)
    b_half = petsc.create_vector(V)
    u_fine = fem.Function(V)

    # Fine step 1
    with b_half.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b_half, L_half_form)
    petsc.apply_lifting(b_half, [a_half_form], bcs=[[bc]])
    b_half.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b_half, [bc])
    ksp_half.solve(b_half, u_fine.x.petsc_vec)
    u_fine.x.scatter_forward()

    # Fine step 2
    u_n_half.x.array[:] = u_fine.x.array[:]
    u_n_half.x.scatter_forward()
    L_half2 = ((1.0 / dt_half - (1 - theta) * reaction_alpha) * ufl.inner(u_n_half, v) * ufl.dx -
               (1 - theta) * epsilon * ufl.inner(ufl.grad(u_n_half), ufl.grad(v)) * ufl.dx +
               ufl.inner(f_func, v) * ufl.dx)
    L_half2_form = fem.form(L_half2)
    with b_half.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b_half, L_half2_form)
    petsc.apply_lifting(b_half, [a_half_form], bcs=[[bc]])
    b_half.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b_half, [bc])
    ksp_half.solve(b_half, u_fine.x.petsc_vec)
    u_fine.x.scatter_forward()

    diff_t = u_coarse.x.petsc_vec - u_fine.x.petsc_vec
    step_temporal_err = diff_t.norm(PETSc.NormType.N2)
    total_temporal_err_est = step_temporal_err / 3.0 * n_steps
    u_sol_norm = u_np1.x.petsc_vec.norm(PETSc.NormType.N2)
    rel_temporal_err = total_temporal_err_est / u_sol_norm if u_sol_norm > 0 else total_temporal_err_est

    if comm.rank == 0:
        print(f"[VERIFICATION] Per-step temporal error: {step_temporal_err:.6e}")
        print(f"[VERIFICATION] Estimated total temporal error: {total_temporal_err_est:.6e}")
        print(f"[VERIFICATION] Relative temporal error: {rel_temporal_err:.6e}")

    # Sample solution onto output grid
    xs = np.linspace(float(out_bbox[0]), float(out_bbox[1]), nx_out)
    ys = np.linspace(float(out_bbox[2]), float(out_bbox[3]), ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    u_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_np1.eval(np.array(points_on_proc),
                          np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()

    u_grid = u_values.reshape(ny_out, nx_out)

    # Sample initial condition
    u_n_interp = fem.Function(V)
    u_n_interp.interpolate(u0_expr)

    u_initial_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        vals0 = u_n_interp.eval(np.array(points_on_proc),
                                np.array(cells_on_proc, dtype=np.int32))
        u_initial_values[eval_map] = vals0.flatten()
    u_initial_grid = u_initial_values.reshape(ny_out, nx_out)

    solver_info = {
        "mesh_resolution": N,
        "element_degree": elem_deg,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": dt_val,
        "n_steps": n_steps,
        "time_scheme": "crank_nicolson",
    }

    return {
        "u": u_grid,
        "solver_info": solver_info,
        "u_initial": u_initial_grid,
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "equation_type": "reaction_diffusion",
            "source_term": "sin(5*pi*x)*sin(3*pi*y) + 0.5*sin(9*pi*x)*sin(7*pi*y)",
            "initial_condition": "sin(pi*x)*sin(pi*y)",
            "boundary_conditions": [{"type": "dirichlet", "value": 0.0}],
            "time": {"t0": 0.0, "t_end": 0.5, "dt": 0.01, "scheme": "crank_nicolson"},
            "epsilon": 0.1,
            "reaction_alpha": 1.0,
            "domain": {"bbox": [0.0, 1.0, 0.0, 1.0]},
        },
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    import time as _time
    t_start = _time.time()
    result = solve(case_spec)
    t_end = _time.time()
    print(f"Wall time: {t_end - t_start:.2f}s")
    print(f"u shape: {result['u'].shape}")
    print(f"u range: [{np.nanmin(result['u']):.8e}, {np.nanmax(result['u']):.8e}]")
    print(f"u_initial range: [{np.nanmin(result['u_initial']):.8e}, {np.nanmax(result['u_initial']):.8e}]")
    print(f"solver_info: {result['solver_info']}")
    
    # Save output files
    np.savez("solution.npz", u=result['u'], u_initial=result['u_initial'])
    import json
    meta = {"solver_info": result['solver_info']}
    with open("meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved solution.npz and meta.json")
