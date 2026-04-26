import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    nu = float(coeffs.get("nu", 0.22))
    
    output_spec = case_spec.get("output", {})
    grid = output_spec.get("grid", {})
    nx_out = int(grid.get("nx", 50))
    ny_out = int(grid.get("ny", 50))
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    N = 192
    
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    mixed_el = basix_mixed_element([vel_el, pres_el])
    W = fem.functionspace(msh, mixed_el)
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # Exact solution (numpy callable)
    def u_ex_np(x):
        X, Y = x[0], x[1]
        u1 = X**2 * (1-X)**2 * (1-2*Y)
        u2 = -2*X*(1-X)*(1-2*X) * Y*(1-Y)
        return np.vstack([u1, u2])
    
    # Source term f = (u·∇)u - νΔu + ∇p (numpy, analytically derived)
    def f_np(x):
        X, Y = x[0], x[1]
        u1 = X**2 * (1-X)**2 * (1-2*Y)
        u2 = -2*X*(1-X)*(1-2*X) * Y*(1-Y)
        du1_dx = 2*X*(1-X)*(1-2*X)*(1-2*Y)
        du1_dy = -2*X**2*(1-X)**2
        du2_dx = -2*(1 - 6*X + 6*X**2) * Y*(1-Y)
        du2_dy = -2*X*(1-X)*(1-2*X)*(1-2*Y)
        d2u1_dx2 = 2*(1-6*X+6*X**2)*(1-2*Y)
        lap_u1 = d2u1_dx2
        d2u2_dx2 = 12*(1-2*X)*Y*(1-Y)
        d2u2_dy2 = 4*X*(1-X)*(1-2*X)
        lap_u2 = d2u2_dx2 + d2u2_dy2
        conv1 = u1*du1_dx + u2*du1_dy
        conv2 = u1*du2_dx + u2*du2_dy
        f1 = conv1 - nu*lap_u1 + 1.0
        f2 = conv2 - nu*lap_u2 + 1.0
        return np.vstack([f1, f2])
    
    # Interpolate f into high-order space
    V_f = fem.functionspace(msh, ("Lagrange", 4, (gdim,)))
    f_func = fem.Function(V_f)
    f_func.interpolate(f_np)
    
    # BCs
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(u_ex_np)
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    bcs = [bc_u]
    
    # Pressure pin at (0,0)
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs[0]) > 0:
        p0 = fem.Function(Q)
        p0.x.array[:] = 0.0
        bcs.append(fem.dirichletbc(p0, p_dofs, W.sub(1)))
    
    # Picard iteration
    w = fem.Function(W)
    u_conv = fem.Function(V)
    u_conv.interpolate(u_ex_np)
    
    (u_t, p_t) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    a_pic = (2*nu*ufl.inner(ufl.sym(ufl.grad(u_t)), ufl.sym(ufl.grad(v)))
             + ufl.inner(ufl.grad(u_t) * u_conv, v)
             - p_t*ufl.div(v) + ufl.div(u_t)*q) * ufl.dx
    L_pic = ufl.inner(f_func, v) * ufl.dx
    
    a_form = fem.form(a_pic)
    L_form = fem.form(L_pic)
    
    A = petsc.create_matrix(a_form)
    b = petsc.create_vector(L_form.function_spaces)
    
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.setFromOptions()
    ksp.setTolerances(rtol=1e-12, atol=1e-14)
    
    w.x.array[:] = 0.0
    total_ksp_its = 0
    picard_its = 0
    picard_tol = 1e-10
    
    for pic_it in range(20):
        u_conv.interpolate(w.sub(0))
        
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=bcs)
        A.assemble()
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        ksp.solve(b, w.x.petsc_vec)
        w.x.scatter_forward()
        
        ksp_its = ksp.getIterationNumber()
        total_ksp_its += ksp_its
        
        if pic_it > 0:
            rel_err = comm.allreduce(
                np.linalg.norm(w.x.array - w_prev) / max(np.linalg.norm(w_prev), 1e-14),
                op=MPI.MAX
            )
            if rel_err < picard_tol:
                picard_its = pic_it + 1
                break
        w_prev = w.x.array.copy()
    else:
        picard_its = 20
    
    ksp.destroy()
    A.destroy()
    b.destroy()
    
    # Extract velocity
    u_h = w.sub(0).collapse()
    
    # Sample velocity magnitude on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(msh, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_mag_values = np.full((pts.shape[1],), np.nan)
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        vals = u_h.eval(pts_arr, cells_arr)
        magnitudes = np.sqrt(vals[:, 0]**2 + vals[:, 1]**2)
        u_mag_values[eval_map] = magnitudes
    
    if comm.size > 1:
        all_values = comm.allgather(u_mag_values)
        combined = np.full_like(u_mag_values, np.nan)
        for vals in all_values:
            mask = ~np.isnan(vals)
            combined[mask] = vals[mask]
        u_mag_values = combined
    
    u_grid = u_mag_values.reshape(ny_out, nx_out)
    
    # Accuracy verification
    u_ex_func = fem.Function(V)
    u_ex_func.interpolate(u_ex_np)
    
    l2_form = fem.form(ufl.inner(u_h - u_ex_func, u_h - u_ex_func) * ufl.dx)
    l2_sq = fem.assemble_scalar(l2_form)
    l2_err = np.sqrt(comm.allreduce(float(l2_sq), op=MPI.SUM))
    
    h1_form = fem.form(ufl.inner(ufl.grad(u_h - u_ex_func), ufl.grad(u_h - u_ex_func)) * ufl.dx)
    h1_sq = fem.assemble_scalar(h1_form)
    h1_err = np.sqrt(comm.allreduce(float(h1_sq), op=MPI.SUM))
    
    exact_mag = np.sqrt(
        (XX**2*(1-XX)**2*(1-2*YY))**2 +
        (-2*XX*(1-XX)*(1-2*XX)*YY*(1-YY))**2
    )
    max_grid_err = np.nanmax(np.abs(u_grid - exact_mag))
    
    if comm.rank == 0:
        print(f"=== Accuracy Verification ===")
        print(f"L2 velocity error: {l2_err:.6e}")
        print(f"H1-semi velocity error: {h1_err:.6e}")
        print(f"Max grid velocity magnitude error: {max_grid_err:.6e}")
        print(f"Mesh: {N}x{N}, Element: Taylor-Hood P2/P1")
        print(f"Picard iterations: {picard_its}, Total KSP iterations: {total_ksp_its}")
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": 2,
            "ksp_type": "preonly",
            "pc_type": "lu",
            "rtol": 1e-12,
            "iterations": total_ksp_its,
            "nonlinear_iterations": [picard_its],
        }
    }
