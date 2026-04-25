import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    t0 = 0.0
    t_end = 0.1
    dt = 0.005
    nx = ny = 64
    degree = 2
    
    comm = MPI.COMM_WORLD
    domain = mesh.create_unit_square(comm, nx, ny, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    x = ufl.SpatialCoordinate(domain)
    t_const = fem.Constant(domain, PETSc.ScalarType(t0))
    
    # Exact solution and source term
    u_ex = ufl.exp(-t_const) * ufl.sin(2*ufl.pi*x[0]) * ufl.sin(2*ufl.pi*x[1])
    u_ex_t = - u_ex
    kappa = 1.0 + 0.3 * ufl.sin(6*ufl.pi*x[0]) * ufl.sin(6*ufl.pi*x[1])
    f = u_ex_t - ufl.div(kappa * ufl.grad(u_ex))
    
    # Initial condition
    u_n = fem.Function(V)
    expr_init = fem.Expression(u_ex, V.element.interpolation_points)
    u_n.interpolate(expr_init)
    
    fdim = domain.topology.dim - 1
    domain.topology.create_connectivity(fdim, domain.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dt_const = fem.Constant(domain, PETSc.ScalarType(dt))
    
    F = (u - u_n) / dt_const * v * ufl.dx + kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
    a = ufl.lhs(F)
    L = ufl.rhs(F)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form)
    A.assemble()
    
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.CG)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=1e-8)
    
    u_h = fem.Function(V)
    u_h.x.array[:] = u_n.x.array[:]
    
    # Grid setup for evaluation
    grid_spec = case_spec.get("output", {}).get("grid", {})
    out_nx = grid_spec.get("nx", 50)
    out_ny = grid_spec.get("ny", 50)
    bbox = grid_spec.get("bbox", [0.0, 1.0, 0.0, 1.0])
    
    xs = np.linspace(bbox[0], bbox[1], out_nx)
    ys = np.linspace(bbox[2], bbox[3], out_ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(out_nx * out_ny)]
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, pts)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, pts)
    
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(pts.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(pts[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
            
    points_on_proc = np.array(points_on_proc)
    cells_on_proc = np.array(cells_on_proc, dtype=np.int32)
    
    u_init_vals = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_n.eval(points_on_proc, cells_on_proc)
        u_init_vals[eval_map] = vals.flatten()
    u_initial = u_init_vals.reshape((out_ny, out_nx))
    
    t = t0
    iterations = 0
    n_steps = 0
    
    while t < t_end - 1e-8:
        t += dt
        n_steps += 1
        t_const.value = t
        
        # update BC
        u_bc.interpolate(fem.Expression(u_ex, V.element.interpolation_points))
        bc = fem.dirichletbc(u_bc, boundary_dofs)
        
        with b.localForm() as loc:
            loc.set(0)
        petsc.assemble_vector(b, L_form)
        petsc.apply_lifting(b, [a_form], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b, [bc])
        
        solver.solve(b, u_h.x.petsc_vec)
        u_h.x.scatter_forward()
        iterations += solver.getIterationNumber()
        
        u_n.x.array[:] = u_h.x.array[:]
        
    u_vals = np.full(pts.shape[0], np.nan)
    if len(points_on_proc) > 0:
        vals = u_h.eval(points_on_proc, cells_on_proc)
        u_vals[eval_map] = vals.flatten()
    u_out = u_vals.reshape((out_ny, out_nx))
    
    solver_info = {
        "mesh_resolution": nx,
        "element_degree": degree,
        "ksp_type": "cg",
        "pc_type": "ilu",
        "rtol": 1e-8,
        "iterations": iterations,
        "dt": dt,
        "n_steps": n_steps,
        "time_scheme": "backward_euler"
    }
    
    return {
        "u": u_out,
        "u_initial": u_initial,
        "solver_info": solver_info
    }

if __name__ == "__main__":
    case = {"output": {"grid": {"nx": 50, "ny": 50, "bbox": [0,1,0,1]}}}
    res = solve(case)
    print("Max u:", np.nanmax(res["u"]))
