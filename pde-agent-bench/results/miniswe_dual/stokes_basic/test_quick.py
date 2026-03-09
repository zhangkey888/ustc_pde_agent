import numpy as np
import time

# Test different resolutions
for N in [80, 90, 100, 110]:
    from importlib import reload
    import solver as s
    
    case_spec = {
        "pde": {"viscosity": 1.0},
        "output": {"nx": 100, "ny": 100},
    }
    
    # Monkey-patch the resolution
    import types
    original_solve = s.solve
    
    def make_solve(n_val):
        def patched_solve(case_spec):
            # Temporarily modify - we'll just call with different N
            import dolfinx
            return original_solve(case_spec)
        return patched_solve
    
    # Just test current N=100
    if N == 100:
        t0 = time.time()
        result = s.solve(case_spec)
        elapsed = time.time() - t0
        
        nx_out, ny_out = 100, 100
        xg = np.linspace(0, 1, nx_out)
        yg = np.linspace(0, 1, ny_out)
        XX, YY = np.meshgrid(xg, yg, indexing='ij')
        u1_exact = np.pi * np.cos(np.pi * YY) * np.sin(np.pi * XX)
        u2_exact = -np.pi * np.cos(np.pi * XX) * np.sin(np.pi * YY)
        vel_mag_exact = np.sqrt(u1_exact**2 + u2_exact**2)
        
        rms_error = np.sqrt(np.mean((result['u'] - vel_mag_exact)**2))
        max_error = np.max(np.abs(result['u'] - vel_mag_exact))
        rel_l2 = np.sqrt(np.sum((result['u'] - vel_mag_exact)**2) / np.sum(vel_mag_exact**2))
        print(f"N={N}: time={elapsed:.3f}s, rms={rms_error:.2e}, max={max_error:.2e}, rel_l2={rel_l2:.2e}")
        break

print("Done")
