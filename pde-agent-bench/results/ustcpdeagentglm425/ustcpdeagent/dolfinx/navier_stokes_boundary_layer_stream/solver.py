import os
os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    nu = float(case_spec["pde"]["coefficients"]["viscosity"])
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    xmin, xmax, ymin, ymax = bbox

    N = 192
    deg_u = 2
    deg_p = 1

    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    fdim = msh.topology.dim - 1

    # Mixed space (Taylor-Hood P2/P1)
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), deg_u, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), deg_p)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    # Exact solution (UFL)
    x = ufl.SpatialCoordinate(msh)
    pi = np.pi
    e6 = ufl.exp(6*(x[0]-1))
    u_ex = ufl.as_vector([pi*e6*ufl.cos(pi*x[1]), -6*e6*ufl.sin(pi*x[1])])
    p_ex = ufl.sin(pi*x[0])*ufl.sin(pi*x[1])

    # Source term: f = (u·∇)u - ν∇²u + ∇p
    grad_u_ex = ufl.grad(u_ex)
    f_ufl = grad_u_ex * u_ex - nu * ufl.div(grad_u_ex) + ufl.grad(p_ex)

    # Interpolate f onto continuous P2 vector space
    V_f = fem.functionspace(msh, ("Lagrange", deg_u, (gdim,)))
    f_func = fem.Function(V_f)
    f_func.interpolate(fem.Expression(f_ufl, V_f.element.interpolation_points))

    # Velocity BC function
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points))

    # Boundary facets and velocity BC
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    bc_u_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bcs = [fem.dirichletbc(u_bc_func, bc_u_dofs, W.sub(0))]

    # Pressure pin at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(p_dofs) > 0:
        p0 = fem.Function(Q); p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))

    # ---- Stokes initialization ----
    (u_t, p_t) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    def eps(u_val): return ufl.sym(ufl.grad(u_val))

    a_s = (ufl.inner(2*nu*eps(u_t), eps(v))*ufl.dx
           - p_t*ufl.div(v)*ufl.dx
           + ufl.div(u_t)*q*ufl.dx)
    L_s = ufl.inner(f_func, v)*ufl.dx

    stokes = fem_petsc.LinearProblem(a_s, L_s, bcs=bcs,
        petsc_options={"ksp_type":"preonly","pc_type":"lu",
                       "pc_factor_mat_solver_type":"mumps"},
        petsc_options_prefix="stokes_")
    w_s = stokes.solve()
    stokes_its = stokes.solver.getIterationNumber()

    w = fem.Function(W)
    w.x.array[:] = w_s.x.array[:]
    w.x.scatter_forward()

    # ---- Newton solve ----
    (u, p) = ufl.split(w)
    def sigma(u_val, p_val): return 2*nu*eps(u_val) - p_val*ufl.Identity(gdim)

    F = (ufl.inner(sigma(u, p), eps(v))*ufl.dx
         + ufl.inner(ufl.grad(u)*u, v)*ufl.dx
         - ufl.inner(f_func, v)*ufl.dx
         + ufl.inner(ufl.div(u), q)*ufl.dx)
    J = ufl.derivative(F, w)

    newton_its = 0
    total_ksp = stokes_its

    try:
        ns_problem = fem_petsc.NonlinearProblem(F, w, bcs=bcs, J=J,
            petsc_options_prefix="ns_",
            petsc_options={
                "snes_type":"newtonls","snes_linesearch_type":"bt",
                "snes_rtol":1e-8,"snes_atol":1e-10,"snes_max_it":50,
                "snes_error_if_not_converged": True,
                "ksp_type":"preonly","pc_type":"lu",
                "pc_factor_mat_solver_type":"mumps",
            })
        w_h = ns_problem.solve()
        w.x.array[:] = w_h.x.array[:]
        w.x.scatter_forward()
        snes = ns_problem.solver
        newton_its = snes.getIterationNumber()
        try:
            total_ksp += snes.getKSP().getIterationNumber()
        except:
            pass
    except Exception as e:
        if comm.rank == 0:
            print(f"Newton failed: {e}, using Picard fallback")

        w.x.array[:] = w_s.x.array[:]
        w.x.scatter_forward()

        u_k = fem.Function(V)
        u_k_expr = fem.Expression(ufl.split(w)[0], V.element.interpolation_points)
        u_k.interpolate(u_k_expr)

        a_p = (ufl.inner(2*nu*eps(u_t), eps(v))*ufl.dx
               + ufl.inner(ufl.grad(u_t)*u_k, v)*ufl.dx
               - p_t*ufl.div(v)*ufl.dx
               + ufl.div(u_t)*q*ufl.dx)

        picard = fem_petsc.LinearProblem(a_p, L_s, bcs=bcs,
            petsc_options={"ksp_type":"preonly","pc_type":"lu",
                           "pc_factor_mat_solver_type":"mumps"},
            petsc_options_prefix="picard_")

        max_picard = 80
        picard_tol = 1e-9
        picard_its = 0

        for it in range(max_picard):
            w_new = picard.solve()
            w_new.x.scatter_forward()
            ksp_it = picard.solver.getIterationNumber()
            total_ksp += ksp_it

            diff = w_new.x.petsc_vec - w.x.petsc_vec
            diff_n = diff.norm(PETSc.NormType.NORM_2)
            w_n = w_new.x.petsc_vec.norm(PETSc.NormType.NORM_2)
            rel = diff_n / w_n if w_n > 1e-14 else diff_n

            w.x.array[:] = w_new.x.array[:]
            w.x.scatter_forward()
            u_k.interpolate(u_k_expr)
            picard_its += 1

            if rel < picard_tol:
                break

        newton_its = picard_its

    # ---- Extract & verify ----
    u_h = w.sub(0).collapse()
    p_h = w.sub(1).collapse()

    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    p_ex_func = fem.Function(Q)
    p_ex_func.interpolate(fem.Expression(p_ex, Q.element.interpolation_points))

    eu = u_h - u_ex_func
    L2e_u = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(eu, eu)*ufl.dx)))
    L2n_u = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(u_ex_func, u_ex_func)*ufl.dx)))
    ep = p_h - p_ex_func
    L2e_p = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(ep, ep)*ufl.dx)))
    L2n_p = np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(p_ex_func, p_ex_func)*ufl.dx)))

    if comm.rank == 0:
        print(f"Vel L2 err: {L2e_u:.3e} (rel {L2e_u/L2n_u:.3e})")
        print(f"Prs L2 err: {L2e_p:.3e} (rel {L2e_p/L2n_p:.3e})")
        print(f"Newton/Picard its: {newton_its}")

    # ---- Sample on output grid ----
    xs = np.linspace(xmin, xmax, nx_out, dtype=np.float64)
    ys = np.linspace(ymin, ymax, ny_out, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out*ny_out, dtype=np.float64)])

    bb = geometry.bb_tree(msh, msh.topology.dim)
    cc = geometry.compute_collisions_points(bb, pts)
    cells_c = geometry.compute_colliding_cells(msh, cc, pts)

    ploc, cloc, emap = [], [], []
    for i in range(pts.shape[0]):
        lk = cells_c.links(i)
        if len(lk) > 0:
            ploc.append(pts[i]); cloc.append(lk[0]); emap.append(i)

    uvals = np.full((nx_out*ny_out, gdim), np.nan, dtype=np.float64)
    if ploc:
        v = u_h.eval(np.array(ploc), np.array(cloc, dtype=np.int32))
        uvals[emap] = v

    uglob = np.zeros_like(uvals)
    comm.Allreduce(uvals, uglob, op=MPI.SUM)
    uglob = np.nan_to_num(uglob, nan=0.0)
    mag = np.linalg.norm(uglob, axis=1).reshape(ny_out, nx_out)

    return {
        "u": mag,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": deg_u,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-10,
            "iterations": int(total_ksp),
            "nonlinear_iterations": [int(newton_its)],
        }
    }

if __name__ == "__main__":
    r = solve({"pde":{"coefficients":{"viscosity":0.08}},
               "output":{"grid":{"nx":50,"ny":50,"bbox":[0.,1.,0.,1.]}}})
    print(f"Shape: {r['u'].shape}, info: {r['solver_info']}")
