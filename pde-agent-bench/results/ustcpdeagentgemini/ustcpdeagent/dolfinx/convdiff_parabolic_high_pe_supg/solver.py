import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from petsc4py import PETSc
from dolfinx.fem import petsc
import time

def solve(case_spec: dict) -> dict:
    nx_out = case_spec["output"]["grid"]["nx"]
    ny_out = case_spec["output"]["grid"]["ny"]
    bbox = case_spec["output"]["grid"]["bbox"]
    
    t0 = 0.0
    t_end = 0.08
    dt = 0.01
    
    mesh_res = 128
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    eps = 0.01
    beta_vec = (10.0, 4.0)
    beta = fem.Constant(domain, PETSc.ScalarType(beta_vec))
    
    x = ufl.SpatialCoordinate(domain)
    
    # exact sol: exp(-t) * sin(2*pi*x) * sin(pi*y)
    t_ufl = fem.Constant(domain, PETSc.ScalarType(t0))
    u_exact = ufl.exp(-t_ufl) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1])
    
    # f = u_t - eps*laplacian(u) + beta \cdot grad(u)
    u_t = -u_exact
    lap_u = - ( (2*ufl.pi)**2 + ufl.pi**2 ) * u_exact
    grad_u = ufl.as_vector([
        ufl.exp(-t_ufl) * 2*ufl.pi * ufl.cos(2*ufl.pi*x[0]) * ufl.sin(ufl.pi*x[1]),
        ufl.exp(-t_ufl) * ufl.pi * ufl.sin(2*ufl.pi*x[0]) * ufl.cos(ufl.pi*x[1])
    ])
    f = u_t - eps * lap_u + ufl.dot(beta, grad_u)
    
    # BCs
    def boundary_marker(x_coord):
        return np.logical_or.reduce([
            np.isclose(x_coord[0], 0.0),
            np.isclose(x_coord[0], 1.0),
            np.isclose(x_coord[1], 0.0),
            np.isclose(x_coord[1], 1.0)
        ])
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, boundary_marker)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
    
    u_initial = fem.Function(V)
    u_initial.x.array[:] = u_n.x.array[:]
    
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    
    # Weak form backward euler
    F_std = (u - u_n)/dt_c * v * ufl.dx \
          + eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx \
          + ufl.dot(beta, ufl.grad(u)) * v * ufl.dx \
          - f * v * ufl.dx
    
    # SUPG stabilization
    h = ufl.CellDiameter(domain)
    vnorm = np.linalg.norm(beta_vec)
    tau = 0.5 * h / vnorm
    
    # Strong residual
    res = (u - u_n)/dt_c - eps * ufl.div(ufl.grad(u)) + ufl.dot(beta, ufl.grad(u)) - f
    F_supg = res * tau * ufl.dot(beta, ufl.grad(v)) * ufl.dx
    
    F = F_std + F_supg
    
    a, L = ufl.system(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)
    
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]
    
    t = t0
    iterations = 0
    n_steps = 0
    while t < t_end - 1e-8:
        t += dt
        t_ufl.value = t
        
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points()))
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        
        u_n.x.array[:] = u_h.x.array[:]
        n_steps += 1
        # For PREONLY LU, it's typically 1 iteration or not meaningful to get, but we add it
        iterations += 1
        
    # Sample on grid
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree_obj = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree_obj, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    u_vals = np.full(nx_out * ny_out, np.nan)
    u_initial_vals = np.full(nx_out * ny_out, np.nan)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    
    for i in range(len(pts)):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    if len(points_on_proc) > 0:
        vals = u_h.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_vals[eval_map] = vals.flatten()
        
        vals_initial = u_initial.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_vals[eval_map] = vals_initial.flatten()
    
    u_grid = u_vals.reshape((ny_out, nx_out))
    u_initial_grid = u_initial_vals.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": degree,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "rtol": 1e-8,
        "iterations": iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_grid,
        "solver_info": solver_info
    }
