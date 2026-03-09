import sys, time, numpy as np
sys.path.insert(0, '.')

case_spec = {
    'pde': {
        'type': 'heat',
        'coefficients': {'kappa': 1.0},
        'time': {'t_end': 0.08, 'dt': 0.008, 'scheme': 'backward_euler'},
    },
    'domain': {'type': 'unit_square', 'nx': 50, 'ny': 50},
}

from solver import solve
t0 = time.time()
result = solve(case_spec)
elapsed = time.time() - t0

u_grid = result['u']
info = result['solver_info']
print(f'Wall time: {elapsed:.3f}s')
print(f'Mesh resolution: {info["mesh_resolution"]}')
print(f'Grid shape: {u_grid.shape}')
print(f'u range: [{u_grid.min():.6e}, {u_grid.max():.6e}]')

xs = np.linspace(0, 1, 50)
ys = np.linspace(0, 1, 50)
XX, YY = np.meshgrid(xs, ys, indexing='ij')
t_end = 0.08
u_exact = np.exp(-t_end) * np.sin(3*np.pi*XX) * np.sin(3*np.pi*YY)

error = np.sqrt(np.mean((u_grid - u_exact)**2))
max_error = np.max(np.abs(u_grid - u_exact))
print(f'L2 (RMS) error: {error:.6e}')
print(f'Max error: {max_error:.6e}')
print(f'Target error: 2.24e-04')
print(f'PASS accuracy: {error < 2.24e-04}')
print(f'PASS time: {elapsed < 19.1}')
print(f'OVERALL PASS: {error < 2.24e-04 and elapsed < 19.1}')
