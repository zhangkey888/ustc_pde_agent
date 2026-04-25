import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    # Grid parameters
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_spec.get("nx", 64)
    ny_out = grid_spec.get("ny", 64)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Mesh parameters
    nx, ny = 64, 64
    domain = mesh.create_rectangle(comm, [[bbox[0], bbox[2]], [bbox[1], bbox[3]]], [nx, ny], cell_type=mesh.CellType.triangle)
    
    V = fem.functionspace(domain, ("Lagrange", 2))
    
    # Time parameters
    t0 = 0.0
    t_end = case_spec.get("t_end", 0.4)
    dt_val = case_spec.get("dt", 0.01)
    epsilon_val = case_spec.get("epsilon", 1.0)
    
    dt = fem.Constant(domain, PETSc.ScalarType(dt_val))
    t = fem.Constant(domain, PETSc.ScalarType(t0))
    epsilon = fem.Constant(domain, PETSc.ScalarType(epsilon_val))
    
    x = ufl.SpatialCoordinate(domain)
    
    # Manufactured solution
    u_ex = ufl.exp(-t) * ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    
    # Source term f = du/dt - eps * div(grad(u))
    # du/dt = -u_ex
    f = -u_ex - epsilon * ufl.div(ufl.grad(u_ex))
    
    # Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points()))
    bc = fem.dirichletbc(u_bc, dofs)
    
    # Initial condition
    u_n = fem.Function(V)
    t.value = t0
    u_n.interpolate(fem.Expression(u_ex, V.element.interpolation_points()))
    
    u_initial = np.zeros((ny_out, nx_out))
    
    # Variational problem (Backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    F = u * v * ufl.dx - u_n * v * ufl.dx + dt * epsilon * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - dt * f * v * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    # Setup probe points
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack((XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())))
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points.T)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[1]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points.T[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    if len(points_on_proc) > 0:
        vals = u_n.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_initial_flat = np.full((points.shape[1],), np.nan)
        u_initial_flat[eval_map] = vals.flatten()
        u_initial = u_initial_flat.reshape((ny_out, nx_out))
        
    t_current = t0
    n_steps = 0
    total_iterations = 0
    
    while t_current < t_end - 1e-8:
        t_current += dt_val
        t.value = t_current
        
        u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points()))
        
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
        n_steps += 1

    u_out = np.zeros((ny_out, nx_out))
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_out_flat = np.full((points.shape[1],), np.nan)
        u_out_flat[eval_map] = vals.flatten()
        u_out = u_out_flat.reshape((ny_out, nx_out))
        
    return {
        "u": u_out,
        "u_initial": u_initial,
        "solver_info": {
            "mesh_resolution": nx,
            "element_degree": 2,
            "ksp_type": "cg",
            "pc_type": "ilu",
            "rtol": 1e-8,
            "iterations": total_iterations,
            "dt": dt_val,
            "n_steps": n_steps,
            "time_scheme": "backward_euler"
        }
    }
