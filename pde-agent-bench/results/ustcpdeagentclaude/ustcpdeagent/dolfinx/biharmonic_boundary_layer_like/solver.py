import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

def solve(case_spec):
    grid = case_spec["output"]["grid"]
    nx, ny = grid["nx"], grid["ny"]
    bbox = grid["bbox"]

    N = 24
    deg = 3
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)

    V = fem.functionspace(msh, ("Lagrange", deg))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.exp(5*(x[0]-1)) * ufl.sin(ufl.pi * x[1])
    lap = ufl.div(ufl.grad(u_exact))
    f = ufl.div(ufl.grad(lap))

    # C0 interior penalty method (Brenner-Sung)
    n = ufl.FacetNormal(msh)
    h = ufl.CellDiameter(msh)
    h_avg = (h('+') + h('-')) / 2.0
    alpha = fem.Constant(msh, PETSc.ScalarType(8.0))

    def jump_grad_n(w):
        return ufl.dot(ufl.grad(w)('+'), n('+')) + ufl.dot(ufl.grad(w)('-'), n('-'))

    def avg_lap(w):
        return 0.5*(ufl.div(ufl.grad(w))('+') + ufl.div(ufl.grad(w))('-'))

    a = ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v)))*ufl.dx \
        - ufl.inner(avg_lap(u), jump_grad_n(v))*ufl.dS \
        - ufl.inner(jump_grad_n(u), avg_lap(v))*ufl.dS \
        + (alpha/h_avg)*ufl.inner(jump_grad_n(u), jump_grad_n(v))*ufl.dS

    # Boundary terms for normal-derivative (natural / weakly imposed via penalty on boundary)
    # For pure Dirichlet u=g we also need to handle ∂u/∂n weakly; simplest: also penalize on boundary
    a += - ufl.inner(ufl.div(ufl.grad(u)), ufl.dot(ufl.grad(v), n))*ufl.ds \
         - ufl.inner(ufl.dot(ufl.grad(u), n), ufl.div(ufl.grad(v)))*ufl.ds \
         + (alpha/h)*ufl.inner(ufl.dot(ufl.grad(u), n), ufl.dot(ufl.grad(v), n))*ufl.ds

    # RHS: include boundary terms for ∂u/∂n from exact solution (since we don't have natural ∂u/∂n=0)
    gN = ufl.dot(ufl.grad(u_exact), n)
    L = ufl.inner(f, v)*ufl.dx \
        - ufl.inner(gN, ufl.div(ufl.grad(v)))*ufl.ds \
        + (alpha/h)*ufl.inner(gN, ufl.dot(ufl.grad(v), n))*ufl.ds

    # Dirichlet BC on u itself
    fdim = msh.topology.dim - 1
    bfacets = mesh.locate_entities_boundary(msh, fdim, lambda x: np.ones(x.shape[1], dtype=bool))
    u_bc = fem.Function(V)
    u_bc.interpolate(fem.Expression(u_exact, V.element.interpolation_points))
    dofs = fem.locate_dofs_topological(V, fdim, bfacets)
    bc = fem.dirichletbc(u_bc, dofs)

    problem = petsc.LinearProblem(a, L, bcs=[bc],
                                   petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
                                   petsc_options_prefix="bih_")
    u_sol = problem.solve()

    xs = np.linspace(bbox[0], bbox[1], nx)
    ys = np.linspace(bbox[2], bbox[3], ny)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.c_[XX.ravel(), YY.ravel(), np.zeros(nx*ny)]

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = []
    for i in range(pts.shape[0]):
        cells.append(coll.links(i)[0])
    vals = u_sol.eval(pts, np.array(cells, dtype=np.int32)).reshape(ny, nx)

    return {
        "u": vals,
        "solver_info": {
            "mesh_resolution": N, "element_degree": deg,
            "ksp_type": "preonly", "pc_type": "lu", "rtol": 1e-12,
            "iterations": 1,
        }
    }

if __name__ == "__main__":
    import time
    spec = {"output": {"grid": {"nx": 64, "ny": 64, "bbox": [0,1,0,1]}}}
    t0 = time.time()
    out = solve(spec)
    t1 = time.time()
    print("time", t1-t0)
    xs = np.linspace(0,1,64); ys = np.linspace(0,1,64)
    XX, YY = np.meshgrid(xs, ys)
    uex = np.exp(5*(XX-1))*np.sin(np.pi*YY)
    err = np.sqrt(np.mean((out["u"]-uex)**2))
    print("rmse", err, "max", np.max(np.abs(out["u"]-uex)))
