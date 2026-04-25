import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec: dict) -> dict:
    pde = case_spec["pde"]
    epsilon = float(pde["parameters"]["epsilon"])
    beta_val = [float(b) for b in pde["parameters"]["beta"]]
    
    output_spec = case_spec["output"]
    grid = output_spec["grid"]
    nx_out = int(grid["nx"])
    ny_out = int(grid["ny"])
    bbox = grid["bbox"]
    
    N = 128
    degree = 2
    rtol = 1e-10
    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", degree))
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    
    beta = ufl.as_vector(beta_val)
    beta_norm = np.sqrt(beta_val[0]**2 + beta_val[1]**2)
    
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = ufl.dot(beta, ufl.grad(u_exact))
    
    h = ufl.CellDiameter(domain)
    tau = h / (2.0 * beta_norm)
    
    a = (epsilon * ufl.inner(ufl.grad(u), ufl.grad(v))
         + ufl.dot(beta, ufl.grad(u)) * v
         + tau * ufl.dot(beta, ufl.grad(v)) * ufl.dot(beta, ufl.grad(u))) * ufl.dx
    
    L = (f * v + tau * ufl.dot(beta, ufl.grad(v)) * f) * ufl.dx
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.ones(x.shape[1], dtype=bool)
    )
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    u_bc = fem.Function(V)
    u_bc.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = fem.dirichletbc(u_bc, boundary_dofs)
    
    a_form = fem.form(a)
    L_form = fem.form(L)
    
    A = petsc.assemble_matrix(a_form, bcs=[bc])
    A.assemble()
    b = petsc.create_vector(L_form.function_spaces)
    
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.GMRES)
    solver.getPC().setType(PETSc.PC.Type.ILU)
    solver.setTolerances(rtol=rtol)
    solver.setFromOptions()
    
    u_sol = fem.Function(V)
    
    with b.localForm() as loc:
        loc.set(0)
    petsc.assemble_vector(b, L_form)
    petsc.apply_lifting(b, [a_form], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    petsc.set_bc(b, [bc])
    
    solver.solve(b, u_sol.x.petsc_vec)
    u_sol.x.scatter_forward()
    
    iterations = solver.getIterationNumber()
    
    L2_error_sq = fem.assemble_scalar(fem.form((u_sol - u_exact)**2 * ufl.dx))
    L2_error = np.sqrt(domain.comm.allreduce(float(L2_error_sq), op=MPI.SUM))
    if domain.comm.rank == 0:
        print(f"L2 error: {L2_error:.6e}, iterations: {iterations}")
    
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx_out)
    ys = np.linspace(ymin, ymax, ny_out)
    XX, YY = np.meshgrid(xs, ys)
    points = np.stack([XX.ravel(), YY.ravel(), np.zeros(nx_out * ny_out)], axis=1)
    
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    
    u_values = np.zeros(nx_out * ny_out)
    points_on_proc = []
    cells_on_proc = []
    eval_map = []
    for i in range(points.shape[0]):
        links = colliding_cells.links(i)
        if len(links) > 0:
            points_on_proc.append(points[i])
            cells_on_proc.append(links[0])
            eval_map.append(i)
    
    if len(points_on_proc) > 0:
        vals = u_sol.eval(
            np.array(points_on_proc),
            np.array(cells_on_proc, dtype=np.int32)
        )
        u_values[eval_map] = vals.flatten()
    
    u_values_global = np.zeros_like(u_values)
    domain.comm.Allreduce(u_values, u_values_global, op=MPI.SUM)
    u_grid = u_values_global.reshape(ny_out, nx_out)
    
    return {
        "u": u_grid,
        "solver_info": {
            "mesh_resolution": N,
            "element_degree": degree,
            "ksp_type": "gmres",
            "pc_type": "ilu",
            "rtol": rtol,
            "iterations": int(iterations),
        }
    }

if __name__ == "__main__":
    case_spec = {
        "pde": {"type": "convection_diffusion", "parameters": {"epsilon": 0.0, "beta": [10.0, 4.0]}, "time": None},
        "output": {"grid": {"nx": 64, "ny": 64, "bbox": [0.0, 1.0, 0.0, 1.0]}}
    }
    import time
    t0 = time.time()
    result = solve(case_spec)
    t1 = time.time()
    print(f"Wall time: {t1-t0:.2f}s")
    print(f"Output shape: {result['u'].shape}")
    print(f"Max value: {np.max(result['u']):.6e}")
    print(f"Solver info: {result['solver_info']}")
