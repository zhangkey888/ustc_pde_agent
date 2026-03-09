import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa = coeffs.get("kappa", 1.0)
    
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.1)
    dt_suggested = time_params.get("dt", 0.005)
    scheme = time_params.get("scheme", "backward_euler")
    
    output = case_spec.get("output", {})
    nx_out = output.get("nx", 50)
    ny_out = output.get("ny", 50)
    
    element_degree = 2
    N = 64
    dt = dt_suggested
    n_steps = int(round(t_end / dt))
    dt = t_end / n_steps
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    
    f_ufl = ufl.exp(-t_const) * ufl.sin(4*ufl.pi*x[0]) * ufl.sin(4*ufl.pi*x[1]) * (-1.0 + 32.0*kappa*ufl.pi**2)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.sin(4*np.pi*X[0]) * np.sin(4*np.pi*X[1]))
    
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    dt_const = fem.Constant(domain, ScalarType(dt))
    kappa_const = fem.Constant(domain, ScalarType(kappa))
    
    a = (u*v/dt_const)*ufl.dx + kappa_const*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L_ufl = (u_n*v/dt_const)*ufl.dx + f_ufl*v*ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    a_form = fem.form(a)
    L_form = fem.form(L_ufl)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    # Create vector using the function space, not the form
    b = petsc.create_vector(V)
    
    u_sol = fem.Function(V)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    total_iterations = 0
    t = 0.0
    for step in range(n_steps):
        t += dt
        t_const.value = t
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
    
    # Evaluate on output grid
    x_out = np.linspace(0, 1, nx_out)
    y_out = np.linspace(0, 1, ny_out)
    X_out, Y_out = np.meshgrid(x_out, y_out, indexing='ij')
    points_3d = np.zeros((nx_out*ny_out, 3))
    points_3d[:, 0] = X_out.ravel()
    points_3d[:, 1] = Y_out.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_grid = np.full(points_3d.shape[0], np.nan)
    u_init_grid = np.full(points_3d.shape[0], np.nan)
    plist, clist, emap = [], [], []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            plist.append(points_3d[i])
            clist.append(links[0])
            emap.append(i)
    
    if plist:
        pts_arr = np.array(plist)
        cells_arr = np.array(clist, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_grid[emap] = vals.flatten()
        vals_init = u_initial_func.eval(pts_arr, cells_arr)
        u_init_grid[emap] = vals_init.flatten()
    
    u_grid = u_grid.reshape((nx_out, ny_out))
    u_init_grid = u_init_grid.reshape((nx_out, ny_out))
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    return {
        "u": u_grid,
        "u_initial": u_init_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": element_degree,
            "ksp_type": "cg",
            "pc_type": "hypre",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }


if __name__ == "__main__":
    case_spec = {
        "pde": {
            "type": "heat",
            "coefficients": {"kappa": 1.0},
            "time": {"t_end": 0.1, "dt": 0.005, "scheme": "backward_euler"}
        },
        "domain": {"type": "unit_square"},
        "output": {"nx": 50, "ny": 50}
    }
    
    start = time.time()
    result = solve(case_spec)
    elapsed = time.time() - start
    
    u_grid = result["u"]
    t_end = 0.1
    x_out = np.linspace(0, 1, 50)
    y_out = np.linspace(0, 1, 50)
    X_out, Y_out = np.meshgrid(x_out, y_out, indexing='ij')
    u_exact = np.exp(-t_end) * np.sin(4*np.pi*X_out) * np.sin(4*np.pi*Y_out)
    
    error = np.sqrt(np.mean((u_grid - u_exact)**2))
    linf_error = np.max(np.abs(u_grid - u_exact))
    print(f"Time: {elapsed:.3f}s")
    print(f"L2 error: {error:.6e}")
    print(f"Linf error: {linf_error:.6e}")
    print(f"N={result['solver_info']['mesh_resolution']}, deg={result['solver_info']['element_degree']}")
    print(f"dt={result['solver_info']['dt']}, n_steps={result['solver_info']['n_steps']}")
