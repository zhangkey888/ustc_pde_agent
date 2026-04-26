import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    
    nu = case_spec["pde"]["params"]["nu"]
    bbox = case_spec["output"]["grid"]["bbox"]
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]

    comm = MPI.COMM_WORLD
    mesh_res = 128

    msh = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    tdim = msh.topology.dim
    fdim = tdim - 1

    # Taylor-Hood P2/P1
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 2, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 1)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)

    pi_c = np.pi
    x = ufl.SpatialCoordinate(msh)

    u_ex = ufl.as_vector([pi_c * ufl.cos(pi_c * x[1]) * ufl.sin(pi_c * x[0]),
                          -pi_c * ufl.cos(pi_c * x[0]) * ufl.sin(pi_c * x[1])])
    p_ex = ufl.cos(pi_c * x[0]) * ufl.cos(pi_c * x[1])

    f = -nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - p * ufl.div(v) * ufl.dx
         + ufl.div(u) * q * ufl.dx)
    L = ufl.inner(f, v) * ufl.dx

    # Velocity BCs on entire boundary
    boundary_facets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    boundary_dofs = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc, boundary_dofs, W.sub(0))
    bcs = [bc_u]

    # Pressure pin at origin
    p_dofs = fem.locate_dofs_geometrical((W.sub(1), Q), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
    if len(p_dofs) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)

    # Manual assembly
    a_form = fem.form(a)
    L_form = fem.form(L)

    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()

    b = petsc.assemble_vector(L_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.apply_lifting(b, [a_form], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, bcs)

    # Solve with LU (umfpack)
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("umfpack")
    ksp.setFromOptions()

    w_h = fem.Function(W)
    ksp.solve(b, w_h.x.petsc_vec)
    w_h.x.scatter_forward()
    iterations = ksp.getIterationNumber()

    u_h = w_h.sub(0).collapse()

    # L2 velocity error for verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
    error_u = fem.form(ufl.inner(u_h - u_exact_func, u_h - u_exact_func) * ufl.dx)
    l2_u = np.sqrt(comm.allreduce(fem.assemble_scalar(error_u), op=MPI.SUM))

    # Sample velocity magnitude on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]).T

    bb_tree = geometry.bb_tree(msh, tdim)
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

    u_mag = np.sqrt(u_values[:, 0]**2 + u_values[:, 1]**2)
    u_grid = u_mag.reshape(ny_out, nx_out)

    # Handle parallel: gather valid values
    if comm.size > 1:
        u_grid_flat = u_grid.ravel()
        u_grid_flat_no_nan = np.where(np.isnan(u_grid_flat), np.inf, u_grid_flat)
        u_grid_global = np.zeros_like(u_grid_flat_no_nan)
        comm.Allreduce(u_grid_flat_no_nan, u_grid_global, op=MPI.MIN)
        u_grid = u_grid_global.reshape(ny_out, nx_out)

    ksp.destroy()
    A.destroy()
    b.destroy()

    t1 = time.time()
    if comm.rank == 0:
        print(f"Time: {t1-t0:.2f}s, L2 vel error: {l2_u:.6e}")

    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": 2,
            "ksp_type": ksp_type,
            "pc_type": pc_type,
            "rtol": rtol,
            "iterations": iterations,
        }
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {"params": {"nu": 0.1}, "time": None},
        "output": {"grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    result = solve(case_spec)
    if MPI.COMM_WORLD.rank == 0:
        print(f"Output shape: {result['u'].shape}, Max: {np.nanmax(result['u']):.6e}")
