import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

ScalarType = PETSc.ScalarType


def solve(case_spec: dict) -> dict:
    pde = case_spec.get("pde", {})
    params = pde.get("parameters", {})
    epsilon = params.get("epsilon", 0.1)
    beta_vec = params.get("beta", [1.0, 0.5])
    
    time_params = pde.get("time", {})
    t_end = time_params.get("t_end", 0.1)
    dt = time_params.get("dt", 0.02)
    scheme = time_params.get("scheme", "backward_euler")
    
    N = 64
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, ScalarType(0.0))
    pi = ufl.pi
    beta = ufl.as_vector([ScalarType(beta_vec[0]), ScalarType(beta_vec[1])])
    
    f_ufl = (-ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
             + epsilon * 2.0 * pi**2 * ufl.exp(-t_const) * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
             + ufl.exp(-t_const) * (beta_vec[0] * pi * ufl.cos(pi * x[0]) * ufl.sin(pi * x[1])
                                    + beta_vec[1] * pi * ufl.sin(pi * x[0]) * ufl.cos(pi * x[1])))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    u_n.interpolate(lambda X: np.sin(np.pi * X[0]) * np.sin(np.pi * X[1]))
    
    u_initial_func = fem.Function(V)
    u_initial_func.x.array[:] = u_n.x.array[:]
    
    dt_c = fem.Constant(domain, ScalarType(dt))
    
    h = ufl.CellDiameter(domain)
    beta_mag = ufl.sqrt(ufl.dot(beta, beta))
    Pe_cell = beta_mag * h / (2.0 * epsilon)
    xi = 1.0 / ufl.tanh(Pe_cell + 1e-10) - 1.0 / (Pe_cell + 1e-10)
    tau = h / (2.0 * beta_mag + 1e-10) * xi
    
    a_gal = (ufl.inner(u, v) / dt_c * ufl.dx
             + epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
             + ufl.inner(ufl.dot(beta, ufl.grad(u)), v) * ufl.dx)
    L_gal = (ufl.inner(u_n, v) / dt_c * ufl.dx
             + ufl.inner(f_ufl, v) * ufl.dx)
    
    v_supg = tau * ufl.dot(beta, ufl.grad(v))
    a_supg = (ufl.inner(u / dt_c, v_supg) * ufl.dx
              + ufl.inner(ufl.dot(beta, ufl.grad(u)), v_supg) * ufl.dx)
    L_supg = (ufl.inner(u_n / dt_c, v_supg) * ufl.dx
              + ufl.inner(f_ufl, v_supg) * ufl.dx)
    
    a_form = a_gal + a_supg
    L_form = L_gal + L_supg
    
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda X: np.ones(X.shape[1], dtype=bool))
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc_func = fem.Function(V)
    bc_func.x.array[:] = 0.0
    bc = fem.dirichletbc(bc_func, dofs)
    bcs = [bc]
    
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)
    
    A = petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()
    b = petsc.create_vector(V)
    
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.GMRES)
    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.ILU)
    ksp.setTolerances(rtol=1e-10, atol=1e-12, max_it=2000)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    n_steps = int(np.round(t_end / dt))
    total_iterations = 0
    current_t = 0.0
    
    for step in range(n_steps):
        current_t += dt
        t_const.value = current_t
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_compiled)
        petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, bcs)
        
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        total_iterations += ksp.getIterationNumber()
        u_n.x.array[:] = u_sol.x.array[:]
    
    nx_out, ny_out = 50, 50
    xs = np.linspace(0, 1, nx_out)
    ys = np.linspace(0, 1, ny_out)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    points_3d = np.zeros((nx_out * ny_out, 3))
    points_3d[:, 0] = XX.ravel()
    points_3d[:, 1] = YY.ravel()
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points_3d)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points_3d)
    
    u_values = np.full(points_3d.shape[0], np.nan)
    u_init_values = np.full(points_3d.shape[0], np.nan)
    pts_list = []
    cells_list = []
    idx_list = []
    for i in range(points_3d.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            pts_list.append(points_3d[i])
            cells_list.append(links[0])
            idx_list.append(i)
    
    if len(pts_list) > 0:
        pts_arr = np.array(pts_list)
        cells_arr = np.array(cells_list, dtype=np.int32)
        vals = u_sol.eval(pts_arr, cells_arr)
        u_values[idx_list] = vals.flatten()
        vals2 = u_initial_func.eval(pts_arr, cells_arr)
        u_init_values[idx_list] = vals2.flatten()
    
    u_grid = u_values.reshape((nx_out, ny_out))
    u_initial_grid = u_init_values.reshape((nx_out, ny_out))
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": 1e-10,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler",
        }
    }
