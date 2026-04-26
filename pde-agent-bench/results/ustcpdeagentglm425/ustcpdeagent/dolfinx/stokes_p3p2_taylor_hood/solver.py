import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["GOTO_NUM_THREADS"] = "4"

import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc as fem_petsc
from basix.ufl import element as basix_element, mixed_element as basix_mixed_element
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    nu = 1.0
    N = 240
    
    msh = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # Taylor-Hood P3/P2 elements
    vel_el = basix_element("Lagrange", msh.topology.cell_name(), 3, shape=(gdim,))
    pres_el = basix_element("Lagrange", msh.topology.cell_name(), 2)
    W = fem.functionspace(msh, basix_mixed_element([vel_el, pres_el]))
    
    V, _ = W.sub(0).collapse()
    Q, _ = W.sub(1).collapse()
    
    (u_trial, p_trial) = ufl.TrialFunctions(W)
    (v_test, q_test) = ufl.TestFunctions(W)
    
    x = ufl.SpatialCoordinate(msh)
    
    # Exact velocity
    u_exact = ufl.as_vector([
        ufl.pi * ufl.cos(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[0]),
        -ufl.pi * ufl.cos(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    ])
    
    # Source term: f = -nu*laplacian(u) + grad(p)
    f_source = ufl.as_vector([
        2*ufl.pi**3 * ufl.cos(ufl.pi*x[1]) * ufl.sin(ufl.pi*x[0])
        - ufl.pi * ufl.sin(ufl.pi*x[0]) * ufl.cos(ufl.pi*x[1]),
        -2*ufl.pi**3 * ufl.cos(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
        - ufl.pi * ufl.cos(ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    ])
    
    # Variational forms
    a_form = (
        nu * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * ufl.dx
        - p_trial * ufl.div(v_test) * ufl.dx
        + q_test * ufl.div(u_trial) * ufl.dx
    )
    L_form = ufl.inner(f_source, v_test) * ufl.dx
    
    # Boundary conditions
    fdim = msh.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    
    u_bc_func = fem.Function(V)
    u_bc_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    dofs_u = fem.locate_dofs_topological((W.sub(0), V), fdim, boundary_facets)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))
    
    bcs = [bc_u]
    
    # Pressure pinning at origin
    p_dofs = fem.locate_dofs_geometrical(
        (W.sub(1), Q),
        lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)
    )
    if len(p_dofs[0]) > 0:
        p0_func = fem.Function(Q)
        p0_func.x.array[:] = 0.0
        bc_p = fem.dirichletbc(p0_func, p_dofs, W.sub(1))
        bcs.append(bc_p)
    
    # Compile forms
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    # Assemble
    A = fem_petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()
    
    b = fem_petsc.assemble_vector(L_compiled)
    fem_petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)
    
    # Solve with MUMPS
    ksp_type = "preonly"
    pc_type = "lu"
    rtol = 1e-10
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.LU)
    pc.setFactorSolverType("mumps")
    solver.setFromOptions()
    
    w_h = fem.Function(W)
    solver.solve(b, w_h.x.petsc_vec)
    w_h.x.scatter_forward()
    
    iterations = solver.getIterationNumber()
    
    # Extract velocity
    u_h = w_h.sub(0).collapse()
    u_h.x.scatter_forward()
    
    # L2 error verification
    u_exact_func = fem.Function(V)
    u_exact_func.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    
    error_expr = ufl.inner(u_h - u_exact_func, u_h - u_exact_func) * ufl.dx
    error_form = fem.form(error_expr)
    error_local = fem.assemble_scalar(error_form)
    error_global = np.sqrt(msh.comm.allreduce(error_local, op=MPI.SUM))
    
    if comm.rank == 0:
        print(f"L2 velocity error: {error_global:.6e}", flush=True)
    
    # Sample velocity magnitude on output grid
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    
    pts = np.zeros((nx_out * ny_out, 3))
    pts[:, 0] = XX.ravel()
    pts[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(msh, msh.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, pts)
    
    u_grid = np.full((ny_out, nx_out), 0.0)
    
    points_on_proc = []
    cells_on_proc = []
    point_indices = []
    
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            point_indices.append(i)
    
    if len(points_on_proc) > 0:
        pts_arr = np.array(points_on_proc)
        cells_arr = np.array(cells_on_proc, dtype=np.int32)
        u_vals = u_h.eval(pts_arr, cells_arr)
        mags = np.linalg.norm(u_vals, axis=1)
        for idx, mag in zip(point_indices, mags):
            u_grid[idx // nx_out, idx % nx_out] = mag
    
    # Gather across processes
    u_flat = u_grid.ravel()
    u_global = np.zeros_like(u_flat) if comm.rank == 0 else None
    comm.Reduce(u_flat, u_global, op=MPI.SUM, root=0)
    if comm.rank == 0:
        u_grid = u_global.reshape(ny_out, nx_out)
    u_grid = comm.bcast(u_grid, root=0)
    
    # Cleanup
    solver.destroy()
    A.destroy()
    b.destroy()
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": 3,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": iterations,
    }
    
    return {"u": u_grid, "solver_info": solver_info}


if __name__ == "__main__":
    import time
    case_spec = {
        "output": {"grid": {"nx": 50, "ny": 50, "bbox": [0.0, 1.0, 0.0, 1.0]}},
        "pde": {"time": None}
    }
    t0 = time.perf_counter()
    result = solve(case_spec)
    t1 = time.perf_counter()
    print(f"Wall time: {t1-t0:.3f}s, shape: {result['u'].shape}, max: {np.nanmax(result['u']):.6e}", flush=True)
