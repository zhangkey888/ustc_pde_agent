import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
import ufl
from dolfinx.fem import petsc
from petsc4py import PETSc
import time

def solve(case_spec: dict) -> dict:
    start_time = time.time()
    
    # 1. Parse parameters
    # Time parameters
    t0 = case_spec.get("time", {}).get("t0", 0.0)
    t_end = case_spec.get("time", {}).get("t_end", 0.1)
    dt = case_spec.get("time", {}).get("dt", 0.01) # Default to slightly smaller dt
    
    # Grid parameters
    grid_spec = case_spec.get("output", {}).get("grid", {})
    nx_out = grid_spec.get("nx", 64)
    ny_out = grid_spec.get("ny", 64)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    # Solver configuration
    mesh_resolution = 64
    element_degree = 2
    ksp_type = "cg"
    pc_type = "hypre"
    rtol = 1e-8

    comm = MPI.COMM_WORLD
    
    # 2. Mesh and Function Space
    domain = mesh.create_unit_square(comm, nx=mesh_resolution, ny=mesh_resolution, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_degree))
    
    # 3. Initial Condition & BC
    u_n = fem.Function(V)
    u_n.x.array[:] = 0.0
    
    u_initial = np.zeros((ny_out, nx_out)) # Placeholder for initial solution grid
    
    # Boundary Conditions
    tdim = domain.topology.dim
    fdim = tdim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    x_coords = ufl.SpatialCoordinate(domain)
    bc_expr = ufl.sin(ufl.pi * x_coords[0]) + ufl.cos(ufl.pi * x_coords[1])
    u_bc.interpolate(fem.Expression(bc_expr, V.element.interpolation_points))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    # 4. Variational Problem (Backward Euler)
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    f = fem.Constant(domain, PETSc.ScalarType(1.0))
    kappa = 1.0 + 0.5 * ufl.sin(2 * ufl.pi * x_coords[0]) * ufl.sin(2 * ufl.pi * x_coords[1])
    
    # (u - u_n)/dt - div(kappa * grad(u)) = f
    # -> u*v/dt + kappa*inner(grad(u), grad(v)) = f*v + u_n*v/dt
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    
    a = ufl.inner(u, v) * ufl.dx + dt_c * kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = dt_c * ufl.inner(f, v) * ufl.dx + ufl.inner(u_n, v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(ksp_type)
    solver.getPC().setType(pc_type)
    solver.setTolerances(rtol=rtol)
    
    u_sol = fem.Function(V)
    u_sol.x.array[:] = u_n.x.array[:]
    
    # Probing setup
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)]
    
    tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(tree, pts)
    colliding = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    pts_on_proc = []
    cells = []
    eval_indices = []
    for i, pt in enumerate(pts):
        if len(colliding.links(i)) > 0:
            pts_on_proc.append(pt)
            cells.append(colliding.links(i)[0])
            eval_indices.append(i)
            
    # Sample initial
    if len(pts_on_proc) > 0:
        vals = u_n.eval(np.array(pts_on_proc), cells)
        u_initial.flat[eval_indices] = vals.flatten()
        
    t = t0
    n_steps = int(round((t_end - t0) / dt))
    total_iterations = 0
    
    for step in range(n_steps):
        t += dt
        
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
        
    # Sample final
    u_final = np.zeros((ny_out, nx_out))
    if len(pts_on_proc) > 0:
        vals = u_sol.eval(np.array(pts_on_proc), cells)
        u_final.flat[eval_indices] = vals.flatten()
        
    solver_info = {
        "mesh_resolution": mesh_resolution,
        "element_degree": element_degree,
        "ksp_type": ksp_type,
        "pc_type": pc_type,
        "rtol": rtol,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_final,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case_spec = {
        "time": {"t0": 0.0, "t_end": 0.1, "dt": 0.01},
        "output": {"grid": {"nx": 50, "ny": 50, "bbox": [0, 1, 0, 1]}}
    }
    res = solve(case_spec)
    print("Done.", res["solver_info"])

