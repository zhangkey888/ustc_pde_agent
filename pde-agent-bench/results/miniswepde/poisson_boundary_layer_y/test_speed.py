import solver
import numpy as np
import time
case_spec = {'pde':{'type':'elliptic','coefficients':{'kappa':1.0}}}
start = time.time()
result = solver.solve(case_spec)
end = time.time()
nx, ny = 50, 50
x = np.linspace(0,1,nx)
y = np.linspace(0,1,ny)
X, Y = np.meshgrid(x, y, indexing='ij')
exact = np.exp(6*Y) * np.sin(np.pi*X)
error = np.max(np.abs(result['u'] - exact))
print(f"Time: {end-start:.3f}s, Error: {error:.2e}, Config: N={result['solver_info']['mesh_resolution']}, degree={result['solver_info']['element_degree']}")
