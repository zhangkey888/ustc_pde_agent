import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Extract parameters
    pde = case_spec["pde"]
    nu_val = float(pde["viscosity"])
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    
    # --- Mesh and Function Space ---
    mesh_res = 192
    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    # --- Trial/Test for Stokes ---
    (u_s, p_s) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    
    f = fem.Constant(msh, PETSc.ScalarType((0.0, 0.0)))
    
    a_stokes = (
        2.0 * nu_val * ufl.inner(ufl.sym(ufl.grad(u_s)), ufl.sym(ufl.grad(v))) * ufl.dx
        - p_s * ufl.div(v) * ufl.dx
        + ufl.div(u_s) * q * ufl.dx
    )
    L_stokes = ufl.inner(f, v) * ufl.dx
    
    # --- Boundary conditions ---
    fdim = msh.topology.dim - 1
    
    # Top wall y=1: u = [1.0, 0.0]
    top_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
    top_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, top_facets)
    u_top = fem.Function(V)
    u_top.interpolate(lambda x: np.vstack([np.ones(x.shape[1]), np.zeros(x.shape[1])]))
    bc_top = fem.dirichletbc(u_top, top_dofs, W.sub(0))
    
    # Right wall x=1: u = [0.0, -0.6]
    right_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 1.0))
    right_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, right_facets)
    u_right = fem.Function(V)
    u_right.interpolate(lambda x: np.vstack([np.zeros(x.shape[1]), np.full(x.shape[1], -0.6)]))
    bc_right = fem.dirichletbc(u_right, right_dofs, W.sub(0))
    
    # Left wall x=0: u = [0.0, 0.0]
    left_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[0], 0.0))
    left_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, left_facets)
    u_zero = fem.Function(V)
    u_zero.x.array[:] = 0.0
    bc_left = fem.dirichletbc(u_zero, left_dofs, W.sub(0))
    
    # Bottom wall y=0: u = [0.0, 0.0]
    bottom_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 0.0))
    bottom_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, bottom_facets)
    u_zero2 = fem.Function(V)
    u_zero2.x.array[:] = 0.0
    bc_bottom = fem.dirichletbc(u_zero2, bottom_dofs, W.sub(0))
    
    bcs = [bc_top, bc_right, bc_left, bc_bottom]
    
    # Pressure pin at origin
    p_origin_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    p0_func = fem.Function(Q)
    p0_func.x.array[:] = 0.0
    bc_p = fem.dirichletbc(p0_func, p_origin_dofs, W.sub(1))
    bcs.append(bc_p)
    
    # --- Manual Stokes solve ---
    a_form = fem.form(a_stokes)
    L_form = fem.form(L_stokes)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)
    
    w = fem.Function(W)
    
    ksp_stokes = PETSc.KSP().create(comm)
    ksp_stokes.setOperators(A)
    ksp_stokes.setType('preonly')
    ksp_stokes.getPC().setType('lu')
    ksp_stokes.getPC().setFactorSolverType('mumps')
    ksp_stokes.setFromOptions()
    ksp_stokes.setTolerances(rtol=1e-10, atol=1e-12)
    
    ksp_stokes.solve(b, w.x.petsc_vec)
    w.x.scatter_forward()
    
    stokes_ksp_its = ksp_stokes.getIterationNumber()
    
    # --- Nonlinear NS problem ---
    (u, p) = ufl.split(w)
    
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u, p):
        return 2.0 * nu_val * eps(u) - p * ufl.Identity(gdim)
    
    F = (
        ufl.inner(sigma(u, p), eps(v)) * ufl.dx
        + ufl.inner(ufl.grad(u) * u, v) * ufl.dx
        - ufl.inner(f, v) * ufl.dx
        + ufl.inner(ufl.div(u), q) * ufl.dx
    )
    
    J = ufl.derivative(F, w)
    
    F_form = fem.form(F)
    J_form = fem.form(J)
    
    # Create SNES
    snes = PETSc.SNES().create(comm)
    snes.setOptionsPrefix("ns_")
    
    def form_function(snes, x, F_vec):
        x.copy(w.x.petsc_vec)
        w.x.scatter_forward()
        with F_vec.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(F_vec, F_form)
        petsc.apply_lifting(F_vec, [J_form], bcs=[bcs], x0=[x], alpha=-1.0)
        F_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(F_vec, bcs, x, alpha=-1.0)
    
    def form_jacobian(snes, x, J_mat, P_mat):
        x.copy(w.x.petsc_vec)
        w.x.scatter_forward()
        J_mat.zeroEntries()
        petsc.assemble_matrix(J_mat, J_form, bcs=bcs)
        J_mat.assemble()
        if P_mat != J_mat:
            P_mat.assemble()
    
    snes.setFunction(form_function, b.duplicate())
    
    J_mat = petsc.create_matrix(J_form)
    snes.setJacobian(form_jacobian, J_mat)
    
    snes.setType('newtonls')
    ls = snes.getLineSearch()
    ls.setType(PETSc.SNESLineSearch.Type.BT)
    snes.setTolerances(rtol=1e-10, atol=1e-12, max_it=50)
    snes.setFromOptions()
    
    # KSP for SNES
    ksp_ns = snes.getKSP()
    ksp_ns.setType('gmres')
    ksp_ns.getPC().setType('lu')
    ksp_ns.getPC().setFactorSolverType('mumps')
    ksp_ns.setTolerances(rtol=1e-8)
    ksp_ns.setFromOptions()
    
    snes.solve(None, w.x.petsc_vec)
    w.x.scatter_forward()
    
    newton_its = snes.getIterationNumber()
    linear_its_total = snes.getLinearSolveIterations()
    
    # Clean up PETSc objects
    ksp_stokes.destroy()
    snes.destroy()
    A.destroy()
    b.destroy()
    
    # --- Extract velocity ---
    u_h = w.sub(0).collapse()
    u_h.x.scatter_forward()
    
    # --- Sample onto output grid ---
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)])
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
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
    
    u_values = np.full((nx_out * ny_out, gdim), np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals
    
    # Velocity magnitude
    mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = mag.reshape(ny_out, nx_out)
    
    # Communicate in parallel
    u_grid_global = np.zeros_like(u_grid)
    comm.Allreduce(u_grid, u_grid_global, op=MPI.SUM)
    u_grid_global = np.nan_to_num(u_grid_global, nan=0.0)
    
    # --- Solver info ---
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": 2,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "rtol": 1e-10,
        "iterations": int(stokes_ksp_its + linear_its_total),
        "nonlinear_iterations": [int(newton_its)],
    }
    
    return {"u": u_grid_global, "solver_info": solver_info}
