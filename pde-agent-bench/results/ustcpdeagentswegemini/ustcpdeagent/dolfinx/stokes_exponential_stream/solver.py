import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD

    # ```DIAGNOSIS
    # equation_type: stokes
    # spatial_dim: 2
    # domain_geometry: rectangle
    # unknowns: vector+scalar
    # coupling: saddle_point
    # linearity: linear
    # time_dependence: steady
    # stiffness: N/A
    # dominant_physics: diffusion
    # peclet_or_reynolds: low
    # solution_regularity: smooth
    # bc_type: all_dirichlet
    # special_notes: manufactured_solution
    # ```
    #
    # ```METHOD
    # spatial_method: fem
    # element_or_basis: Taylor-Hood_P2P1
    # stabilization: none
    # time_method: none
    # nonlinear_solver: none
    # linear_solver: preonly
    # preconditioner: lu
    # special_treatment: pressure_pinning
    # pde_skill: stokes
    # ```

    pde = case_spec.get("pde", {})
    output = case_spec.get("output", {})
    grid = output.get("grid", {})
    nx_out = int(grid.get("nx", 64))
    ny_out = int(grid.get("ny", 64))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])

    nu = float(pde.get("nu", pde.get("viscosity", 1.0)))

    # High-accuracy default chosen to satisfy strict manufactured-solution tolerance
    mesh_resolution = int(case_spec.get("mesh_resolution", 128))
    element_degree = 2

    msh = mesh.create_unit_square(
        comm, mesh_resolution, mesh_resolution, cell_type=mesh.CellType.triangle
    )
    gdim = msh.geometry.dim
    cell_name = msh.topology.cell_name()

    vel_el = basix_element("Lagrange", cell_name, 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", cell_name, 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))

    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    x = ufl.SpatialCoordinate(msh)
    pi = ufl.pi

    u_exact_ufl = ufl.as_vector(
        [pi * ufl.exp(x[0]) * ufl.cos(pi * x[1]), -ufl.exp(x[0]) * ufl.sin(pi * x[1])]
    )
    p_exact_ufl = ufl.exp(x[0]) * ufl.cos(pi * x[1])

    f_ufl = -nu * ufl.div(ufl.grad(u_exact_ufl)) + ufl.grad(p_exact_ufl)

    u_trial, p_trial = ufl.TrialFunctions(W)
    v_test, q_test = ufl.TestFunctions(W)

    a = (
        nu * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
        - ufl.inner(p_trial, ufl.div(v_test)) * ufl.dx
        + ufl.inner(ufl.div(u_trial), q_test) * ufl.dx
    )
    L = ufl.inner(f_ufl, v_test) * ufl.dx

    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda X: np.ones(X.shape[1], dtype=bool)
    )

    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, dofs_u, W.sub(0))

    p_pin_fun = fem.Function(Q)
    p_pin_fun.x.array[:] = 0.0
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda X: np.isclose(X[0], 0.0) & np.isclose(X[1], 0.0),
    )
    bcs = [bc_u]
    if len(p_dofs) > 0:
        bc_p = fem.dirichletbc(p_pin_fun, p_dofs, W.sub(1))
        bcs.append(bc_p)

    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1.0e-12

    problem = petsc.LinearProblem(
        a,
        L,
        bcs=bcs,
        petsc_options_prefix="stokes_",
        petsc_options={
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "pc_factor_mat_solver_type": "mumps",
        },
    )

    try:
        wh = problem.solve()
    except Exception:
        problem = petsc.LinearProblem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="stokes_fallback_",
            petsc_options={
                "ksp_type": "minres",
                "pc_type": "lu",
                },
        )
        wh = problem.solve()
        ksp_type = "minres"
        pc_type = "lu"

    wh.x.scatter_forward()

    u_h = wh.sub(0).collapse()
    p_h = wh.sub(1).collapse()

    # Accuracy verification
    u_exact = fem.Function(V)
    u_exact.interpolate(fem.Expression(u_exact_ufl, V.element.interpolation_points))
    err_fun = fem.Function(V)
    err_fun.x.array[:] = u_h.x.array - u_exact.x.array
    err_fun.x.scatter_forward()

    l2_error_local = fem.assemble_scalar(fem.form(ufl.inner(err_fun, err_fun) * ufl.dx))
    l2_error = np.sqrt(comm.allreduce(l2_error_local, op=MPI.SUM))

    div_u_local = fem.assemble_scalar(fem.form((ufl.div(u_h) ** 2) * ufl.dx))
    div_u_l2 = np.sqrt(comm.allreduce(div_u_local, op=MPI.SUM))

    # Grid sampling of velocity magnitude
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)

    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)

    local_vals = np.full((pts.shape[0], gdim), np.nan, dtype=np.float64)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc, dtype=np.float64), np.array(cells_on_proc, dtype=np.int32))
        local_vals[np.array(eval_map, dtype=np.int32), :] = np.real(vals)

    gathered = comm.gather(local_vals, root=0)

    if comm.rank == 0:
        global_vals = np.full((pts.shape[0], gdim), np.nan, dtype=np.float64)
        for arr in gathered:
            mask = np.isfinite(arr[:, 0])
            global_vals[mask] = arr[mask]
        mag = np.linalg.norm(global_vals, axis=1).reshape(ny_out, nx_out)
    else:
        mag = None

    mag = comm.bcast(mag, root=0)

    iterations = 1
    try:
        ksp = problem.solver
        iterations = int(ksp.getIterationNumber())
    except Exception:
        pass

    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
        "l2_error_velocity": float(l2_error),
        "divergence_l2": float(div_u_l2),
    }

    return {"u": mag, "solver_info": solver_info}


if __name__ == "__main__":
    case_spec = {
        "pde": {"nu": 1.0, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}},
    }
    out = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(out["u"].shape)
        print(out["solver_info"])
