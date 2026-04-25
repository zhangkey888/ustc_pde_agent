import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    pde = case_spec["pde"]
    coeffs = pde.get("coefficients", {})
    kappa = coeffs.get("kappa", 1.0)
    time_info = pde["time"]
    t0 = time_info["t0"]
    t_end = time_info["t_end"]
    
    out = case_spec["output"]
    grid = out["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    mesh_res = 80
    elem_deg = 2
    n_steps = 60
    dt = (t_end - t0) / n_steps
    time_scheme = "crank_nicolson"
    rtol = 1e-10
    
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    tdim = domain.topology.dim
    fdim = tdim - 1
    
    V = fem.functionspace(domain, ("Lagrange", elem_deg))
    
    u_n = fem.Function(V)
    u_n.interpolate(lambda x: np.sin(6*np.pi*x[0])*np.sin(6*np.pi*x[1]))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    a = u * v * ufl.dx + (dt / 2.0) * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = u_n * v * ufl.dx - (dt / 2.0) * kappa * ufl.inner(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
    
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.HYPRE)
    solver.getPC().setHYPREType("boomeramg")
    solver.setTolerances(rtol=rtol, atol=1e-14, max_it=500)
    solver.setInitialGuessNonzero(True)
    
    u_sol = fem.Function(V)
    total_iterations = 0
    
    for step in range(n_steps):
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
    
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.zeros((nx_out * ny_out, 3))
    points[:, 0] = XX.ravel()
    points[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, tdim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    u_values = np.full((points.shape[0],), np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    if comm.size > 1:
        local_data = np.where(np.isnan(u_values), 0.0, u_values)
        local_count = np.where(np.isnan(u_values), 0.0, 1.0)
        global_sum = np.zeros_like(u_values)
        global_count = np.zeros_like(u_values)
        comm.Allreduce(local_data, global_sum, op=MPI.SUM)
        comm.Allreduce(local_count, global_count, op=MPI.SUM)
        u_values = np.where(global_count > 0, global_sum / global_count, np.nan)
    
    u_grid = u_values.reshape(ny_out, nx_out)
    u_initial_grid = np.sin(6*np.pi*XX)*np.sin(6*np.pi*YY)
    
    exact_factor = np.exp(-72 * np.pi**2 * t_end)
    u_exact_grid = np.sin(6*np.pi*XX)*np.sin(6*np.pi*YY)*exact_factor
    
    mask = ~np.isnan(u_grid)
    if np.any(mask):
        denom = np.sqrt(np.sum(u_exact_grid[mask]**2))
        l2_error = np.sqrt(np.sum((u_grid[mask] - u_exact_grid[mask])**2)) / max(denom, 1e-15)
    else:
        l2_error = float('inf')
    
    if comm.rank == 0:
        print(f"L2_ERROR: {l2_error:.6e}")
        print(f"Mesh: {mesh_res}, Degree: {elem_deg}, dt: {dt:.2e}, Steps: {n_steps}")
        print(f"Total KSP its: {total_iterations}")
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": elem_deg,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": rtol,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": time_scheme,
        },
    }

if __name__ == "__main__":
    import time
    case_spec = {
        "pde": {"type": "heat", "coefficients": {"kappa": 1.0},
                "time": {"t0": 0.0, "t_end": 0.005, "dt": 0.001, "scheme": "backward_euler"},
                "initial_condition": "sin(6*pi*x)*sin(6*pi*y)",
                "boundary_conditions": [{"type": "dirichlet", "value": 0.0}]},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}, "field": "scalar"},
    }
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    print(f"WALL_TIME: {elapsed:.3f}")
    print(f"Shape: {result['u'].shape}")
