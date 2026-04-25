import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, geometry
import ufl

def solve(case_spec: dict) -> dict:
    # 1. Read case_spec
    output_grid = case_spec.get("output", {}).get("grid", {})
    nx_out = output_grid.get("nx", 50)
    ny_out = output_grid.get("ny", 50)
    bbox = output_grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    t0 = 0.0
    t_end = 0.2
    dt = 0.01
    
    mesh_res = 64
    degree = 2
    kappa = 0.5
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    # Time variable
    t = t0
    t_c = fem.Constant(domain, PETSc.ScalarType(t))
    
    # Exact solution expression for IC, BC, and source
    x = ufl.SpatialCoordinate(domain)
    u_ex_t = ufl.exp(-2.0 * t_c) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term f = du/dt - kappa * Delta u
    # du/dt = -2 * u, Delta u = -2 * pi^2 * u
    f_expr = (-2.0 + kappa * 2.0 * ufl.pi**2) * u_ex_t
    
    # Initial condition
    u_n = fem.Function(V)
    u_n.interpolate(fem.Expression(u_ex_t, V.element.interpolation_points()))
    
    def sample_on_grid(u_func):
        xs = np.linspace(bbox[0], bbox[1], nx_out)
        ys = np.linspace(bbox[2], bbox[3], ny_out)
        XX, YY = np.meshgrid(xs, ys)
        pts = np.vstack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())])
        
        bb_tree = geometry.bb_tree(domain, domain.topology.dim)
        cell_candidates = geometry.compute_collisions_points(bb_tree, pts.T)
        colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts.T)
        
        points_on_proc = []
        cells_on_proc = []
        eval_map = []
        for i in range(pts.shape[1]):
            links = colliding_cells.links(i)
            if len(links) > 0:
                points_on_proc.append(pts.T[i])
                cells_on_proc.append(links[0])
                eval_map.append(i)
                
        u_values = np.full((pts.shape[1],), np.nan)
        if len(points_on_proc) > 0:
            vals = u_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
            u_values[eval_map] = vals.flatten()
            
        return u_values.reshape(ny_out, nx_out)

    u_initial = sample_on_grid(u_n)
    
    # Boundary conditions
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    bdofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, bdofs)
    
    # Variational problem (Backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    
    F = (u - u_n) / dt_c * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f_expr * v * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = fem.petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    # Create RHS vector (pass list of function spaces as required by dolfinx 0.10.0)
    b = fem.petsc.create_vector(L_form)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType("cg")
    solver.getPC().setType("jacobi")
    solver.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    
    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0
    
    for i in range(n_steps):
        t += dt
        t_c.value = t
        
        # Update BC
        u_bc.interpolate(fem.Expression(u_ex_t, V.element.interpolation_points()))
        
        with b.localForm() as loc:
            loc.set(0)
        fem.petsc.assemble_vector(b, L_form)
        fem.petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        # Update previous solution
        u_n.x.array[:] = u_sol.x.array
        
    u_final = sample_on_grid(u_sol)
    
    return {
        "u": u_final,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "jacobi",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }
