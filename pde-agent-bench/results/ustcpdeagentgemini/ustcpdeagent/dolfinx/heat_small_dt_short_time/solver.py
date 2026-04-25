import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    comm = MPI.COMM_WORLD
    
    grid = case_spec["output"]["grid"]
    nx_out = grid["nx"]
    ny_out = grid["ny"]
    bbox = grid["bbox"]
    
    mesh_res = 64
    element_deg = 2
    dt = 0.003
    t_end = 0.06
    n_steps = int(round(t_end / dt))
    
    domain = mesh.create_unit_square(comm, nx=mesh_res, ny=mesh_res, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", element_deg))
    
    x = ufl.SpatialCoordinate(domain)
    t = fem.Constant(domain, PETSc.ScalarType(0.0))
    
    u_exact = ufl.exp(-t) * ufl.sin(4 * ufl.pi * x[0]) * ufl.sin(4 * ufl.pi * x[1])
    f = (32.0 * ufl.pi**2 - 1.0) * u_exact
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    u_n = fem.Function(V)
    
    def u_init(x_coords):
        return np.sin(4 * np.pi * x_coords[0]) * np.sin(4 * np.pi * x_coords[1])
    
    u_n.interpolate(u_init)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True))
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    u_bc = fem.Function(V)
    
    dt_c = fem.Constant(domain, PETSc.ScalarType(dt))
    a = ufl.inner(u, v) * ufl.dx + dt_c * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(u_n, v) * ufl.dx + dt_c * ufl.inner(f, v) * ufl.dx
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form)
    A.assemble()
    
    b_func = fem.Function(V)
    b = b_func.x.petsc_vec
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8)
    
    u_sol = fem.Function(V)
    
    total_iterations = 0
    t_val = 0.0
    for i in range(n_steps):
        t_val += dt
        t.value = t_val
        
        u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        A.zeroEntries()
        petsc.assemble_matrix(A, a_form, bcs=[bc])
        A.assemble()
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_sol.x.petsc_vec)
        u_sol.x.scatter_forward()
        
        total_iterations += solver.getIterationNumber()
        
        u_n.x.array[:] = u_sol.x.array
    
    xs = np.linspace(bbox[0], bbox[1], nx_out)
    ys = np.linspace(bbox[2], bbox[3], ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.vstack([XX.ravel(), YY.ravel(), np.zeros_like(XX.ravel())])
    
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
    
    u_values = np.zeros(points.shape[1])
    if len(points_on_proc) > 0:
        vals = u_sol.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_values[eval_map] = vals.flatten()
    
    u_grid = u_values.reshape((ny_out, nx_out))
    
    u_init_func = fem.Function(V)
    u_init_func.interpolate(u_init)
    
    u_init_values = np.zeros(points.shape[1])
    if len(points_on_proc) > 0:
        vals = u_init_func.eval(np.array(points_on_proc), np.array(cells_on_proc, dtype=np.int32))
        u_init_values[eval_map] = vals.flatten()
    u_initial_array = u_init_values.reshape((ny_out, nx_out))
    
    solver_info = {
        "mesh_resolution": mesh_res,
        "element_degree": element_deg,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": total_iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_grid,
        "u_initial": u_initial_array,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case = {
        "output": {
            "grid": {
                "nx": 50,
                "ny": 50,
                "bbox": [0, 1, 0, 1]
            }
        }
    }
    sol = solve(case)
    print("Test finished successfully.")
