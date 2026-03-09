"""Test the biharmonic solver with a manufactured solution to verify correctness."""
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, geometry
from dolfinx.fem import petsc
import ufl
from petsc4py import PETSc

comm = MPI.COMM_WORLD
pi = np.pi

# Manufactured solution for simply supported biharmonic on [0,1]^2:
# u_exact = sin(pi*x)*sin(pi*y)
# Δu = -pi²*sin(pi*x)*sin(pi*y) - pi²*sin(pi*x)*sin(pi*y) = -2*pi²*sin(pi*x)*sin(pi*y)
# Δ²u = Δ(-2*pi²*sin(pi*x)*sin(pi*y)) = -2*pi²*(-2*pi²*sin(pi*x)*sin(pi*y)) = 4*pi⁴*sin(pi*x)*sin(pi*y)
# So f = 4*pi⁴*sin(pi*x)*sin(pi*y)
# BCs: u = 0 on ∂Ω (satisfied), Δu = 0 on ∂Ω (satisfied since sin(0)=sin(pi*1)... wait sin(pi)=0, yes)

N = 64
degree = 2
domain = mesh.create_unit_square(comm, N, N, cell_type=mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", degree))

x = ufl.SpatialCoordinate(domain)
f_expr = 4 * pi**4 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
u_exact_expr = ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])

tdim = domain.topology.dim
fdim = tdim - 1
boundary_facets = mesh.locate_entities_boundary(
    domain, fdim, lambda xc: np.ones(xc.shape[1], dtype=bool)
)

# Step 1: Solve standard Poisson -Δw = -f with w=0 on ∂Ω
# Since Δw = f => -Δw = -f => ∫∇w·∇v dx = ∫(-(-f))v dx... 
# Let me be very careful:
# Δw = f
# Multiply by test v, integrate: ∫Δw v dx = ∫f v dx
# IBP: -∫∇w·∇v dx + boundary = ∫f v dx
# With v∈H¹₀: -∫∇w·∇v dx = ∫f v dx
# So: ∫∇w·∇v dx = -∫f v dx

w_trial = ufl.TrialFunction(V)
v_test = ufl.TestFunction(V)

a1 = ufl.inner(ufl.grad(w_trial), ufl.grad(v_test)) * ufl.dx
L1 = -ufl.inner(f_expr, v_test) * ufl.dx

dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, V)

prob1 = petsc.LinearProblem(a1, L1, bcs=[bc],
    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-12"},
    petsc_options_prefix="test1_")
w_sol = prob1.solve()

# Check w: should be Δu = -2π²sin(πx)sin(πy)
w_exact_expr = -2 * pi**2 * ufl.sin(pi * x[0]) * ufl.sin(pi * x[1])
w_err = fem.form((w_sol - w_exact_expr)**2 * ufl.dx)
w_err_val = np.sqrt(fem.assemble_scalar(w_err))
w_norm = fem.form(w_exact_expr**2 * ufl.dx)
w_norm_val = np.sqrt(fem.assemble_scalar(w_norm))
print(f"w error L2: {w_err_val:.6e}, w norm: {w_norm_val:.6e}, relative: {w_err_val/w_norm_val:.6e}")

# Step 2: Solve Δu = w with u=0 on ∂Ω
# Δu = w => -Δu = -w => ∫∇u·∇v dx = -∫w v dx... wait:
# Δu = w => multiply by v: ∫Δu v dx = ∫w v dx
# IBP: -∫∇u·∇v dx = ∫w v dx
# So: ∫∇u·∇v dx = -∫w v dx

u_trial = ufl.TrialFunction(V)
v_test2 = ufl.TestFunction(V)

a2 = ufl.inner(ufl.grad(u_trial), ufl.grad(v_test2)) * ufl.dx
L2 = -ufl.inner(w_sol, v_test2) * ufl.dx

prob2 = petsc.LinearProblem(a2, L2, bcs=[bc],
    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": "1e-12"},
    petsc_options_prefix="test2_")
u_sol = prob2.solve()

# Check u error
u_err = fem.form((u_sol - u_exact_expr)**2 * ufl.dx)
u_err_val = np.sqrt(fem.assemble_scalar(u_err))
u_norm = fem.form(u_exact_expr**2 * ufl.dx)
u_norm_val = np.sqrt(fem.assemble_scalar(u_norm))
print(f"u error L2: {u_err_val:.6e}, u norm: {u_norm_val:.6e}, relative: {u_err_val/u_norm_val:.6e}")

# Also check max value
u_max_form = fem.form(u_sol * ufl.dx)
print(f"u integral: {fem.assemble_scalar(u_max_form):.6e}")
print(f"Expected integral of sin(πx)sin(πy): {(2/pi)**2:.6e}")
