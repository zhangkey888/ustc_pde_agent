import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    t0 = time.time()
    # Read grid
    grid = case_spec.get("output", {}).get("grid", {})
    nx_out = grid.get("nx", 50)
    ny_out = grid.get("ny", 50)
    bbox = grid.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Time params
    time_spec = case_spec.get("pde", {}).get("time", {})
    t_end = time_spec.get("t_end", 0.6)
    dt = time_spec.get("dt", 0.01)
    
    # Physics params
    epsilon = case_spec.get("pde", {}).get("epsilon", 1.0)
    alpha = case_spec.get("pde", {}).get("reaction_alpha", 1.0)
    
    # Numerical params
    mesh_res = case_spec.get("solver", {}).get("mesh_resolution", 64)
    degree = case_spec.get("solver", {}).get("element_degree", 2)
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, mesh_res, mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    u_n = fem.Function(V)
    u_sol = fem.Function(V)
    
    x = ufl.SpatialCoordinate(domain)
    
    t_ufl = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    def exact_expr(t_val):
        return ufl.exp(-t_val) * ufl.cos(2 * ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    u_exact = exact_expr(t_ufl)
    
    f_expr = (-1.0 + 5.0 * ufl.pi**2 * epsilon + alpha) * u_exact
    
    # Boundary conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Weak form: backward Euler
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    F = ufl.inner((u - u_n)/dt_const, v)*ufl.dx + epsilon*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx + alpha*ufl.inner(u, v)*ufl.dx - ufl.inner(f_expr, v)*ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    # Precompile Expression
    u_exact_expr = fem.Expression(u_exact, V.element.interpolation_points)
    
    # Initialize
    t_ufl.value = 0.0
    u_n.interpolate(u_exact_expr)
    u_sol.x.array[:] = u_n.x.array
    
    # Solver setup
    a_form = fem.form(a)
    L_form = fem.form(L)
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)
    ksp.getPC().setType(PETSc.PC.Type.ILU)
    ksp.setTolerances(rtol=1e-8)
    
    t_val = 0.0
    n_steps = int(np.ceil((t_end - t_val) / dt))
    
    total_iters = 0
    
    for i in range(n_steps):
        t_val += dt
        t_ufl.value = t_val
        
        # update BC
        u_bc.interpolate(u_exact_expr)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        ksp.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        u_n.x.array[:] = u_sol.x.array
        
        total_iters += ksp.getIterationNumber()
        
    # sample output
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i_pt in range(points.shape[0]):
        links = colliding_cells.links(i_pt)
        if len(links) > 0:
            points_on_proc.append(points[i_pt])
            cells_on_proc.append(links[0])
            eval_map.append(i_pt)
            
    u_out = np.full(nx_out * ny_out, np.nan)
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out[eval_map] = vals.flatten()
        
    u_out = u_out.reshape((ny_out, nx_out))
    
    return {
        "u": u_out,
        "solver_info": {
            "mesh_resolution": mesh_res,
            "element_degree": degree,
            "ksp_type": "cg",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_iters,
            "dt": dt,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }
