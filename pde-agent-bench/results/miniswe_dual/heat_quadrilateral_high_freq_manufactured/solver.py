import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time as time_module

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    coeffs = pde.get("coefficients", {})
    kappa_val = float(coeffs.get("kappa", coeffs.get("k", 1.0)))
    
    time_params = pde.get("time", {})
    t_end = float(time_params.get("t_end", 0.1))
    dt_suggested = float(time_params.get("dt", 0.005))
    scheme = time_params.get("scheme", "backward_euler")
    
    output = case_spec.get("output", {})
    nx_out = int(output.get("nx", 50))
    ny_out = int(output.get("ny", 50))
    
    element_degree = 2
    N = 64
    dt = dt_suggested
    
    result = _solve_at_resolution(N, element_degree, kappa_val, t_end, dt, scheme, nx_out, ny_out)
    result["solver_info"]["mesh_resolution"] = N
    return result


def _solve_at_resolution(N, element_degree, kappa_val, t_end, dt, scheme, nx_out, ny_out):
    comm = MPI.COMM_WORLD
    
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.quadrilateral)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, ScalarType(0.0))
    dt_const = fem.Constant(domain, ScalarType(dt))
    kappa = fem.Constant(domain, ScalarType(kappa_val))
    
    pi = ufl.pi
    f_expr = ufl.exp(-t) * ufl.sin(4*pi*x[0]) * ufl.sin(4*pi*x[1]) * (-1.0 + 32.0*kappa_val*pi**2)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V, name="u_n")
    u_n.interpolate(lambda X: np.sin(4*np.pi*X[0]) * np.sin(4*np.pi*X[1]))
    
    a = u*v*ufl.dx + dt_const*kappa*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = u_n*v*ufl.dx + dt_const*f_expr*v*ufl.dx
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0.0), dofs, V)
    bcs = [bc]
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    
    b = A.createVecRight()
    u_sol = fem.Function(V, name="u_sol")
    
    solver = PETSc.KSP().create(comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    pc = solver.getPC()
    pc.setType(PETSc.PC.Type.HYPRE)
    solver.setTolerances(rtol=1e-10, atol=1e-12, max_it=1000)
    solver.setUp()
    
    n_steps = int(np.ceil(t_end / dt))
    actual_dt = t_end / n_steps
    dt_const.value = actual_dt
    
    total_iterations = 0
    current_t = 0.0
    
    for step in range(n_steps):
        current_t += actual_dt
        t.value = current_t
        
        with b.localForm() as loc:
            loc.set(0.0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += solver.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
    
    u_grid = _evaluate_on_grid(domain, u_sol, nx_out, ny_out)
    
    u_n.interpolate(lambda X: np.sin(4*np.pi*X[0]) * np.sin(4*np.pi*X[1]))
    u_initial = _evaluate_on_grid(domain, u_n, nx_out, ny_out)
    
    solver.destroy()
    A.destroy()
    b.destroy()
    
    solver_info = {
        "mesh_resolution": N,
        "element_degree": element_degree,
        "ksp_type": "cg",
        "pc_type": "hypre",
        "rtol": 1e-10,
        "iterations": total_iterations,
        "dt": actual_dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler",
    }
    
    return {"u": u_grid, "u_initial": u_initial, "solver_info": solver_info}


def _evaluate_on_grid(domain, u_func, nx, ny):
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    
    points_2d = np.column_stack([XX.ravel(), YY.ravel()])
    points_3d = np.zeros((points_2d.shape[0], 3))
    points_3d[:, :2] = points_2d
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
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
        pts = np.array(points_on_proc)
        cls = np.array(cells_on_proc, dtype=np.int32)
        vals = u_func.eval(pts, cls)
        u_values[eval_map] = vals.flatten()
    
    return u_values.reshape((nx, ny))
