import re

with open('solver.py', 'r') as f:
    content = f.read()

# Replace the early stopping condition with time-based stopping
old_pattern = r'''        # Broadcast error to all ranks for loop control
        grid_l2_error = comm.bcast(grid_l2_error, root=0)
        
        # Early exit if we've achieved excellent accuracy \(much better than target\)
        # and we're not at the first configuration
        if grid_l2_error < target_error / 10.0 and config_idx > 0:
            if rank == 0:
                print\(f"  Achieved excellent accuracy \({grid_l2_error:.2e} << {target_error:.2e}\), stopping"\)
            break'''

new_text = '''        # Broadcast error to all ranks for loop control
        grid_l2_error = comm.bcast(grid_l2_error, root=0)
        
        # Check time limit - continue refining until we're close to the limit
        elapsed = time.time() - start_time
        if elapsed > 0.95 * time_limit:  # Stop if we've used 95% of time
            if rank == 0:
                print(f"  Time limit approaching ({elapsed:.2f}s), stopping refinement")
            break'''

content = re.sub(re.escape(old_pattern), new_text, content, flags=re.DOTALL)

with open('solver.py', 'w') as f:
    f.write(content)
