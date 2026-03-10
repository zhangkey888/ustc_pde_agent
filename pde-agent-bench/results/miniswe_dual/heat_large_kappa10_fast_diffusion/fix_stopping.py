with open('solver.py', 'r') as f:
    lines = f.readlines()

# Find the section to replace
start_idx = -1
for i in range(len(lines)):
    if "# Broadcast error to all ranks for loop control" in lines[i]:
        start_idx = i
        break

if start_idx != -1:
    # Find the break statement
    end_idx = start_idx
    for i in range(start_idx, len(lines)):
        if "break" in lines[i]:
            end_idx = i
            break
    
    # Replace the section
    new_section = '''        # Broadcast error to all ranks for loop control
        grid_l2_error = comm.bcast(grid_l2_error, root=0)
        
        # Check time limit - continue refining until we're close to the limit
        elapsed = time.time() - start_time
        if elapsed > 0.95 * time_limit:  # Stop if we've used 95% of time
            if rank == 0:
                print(f"  Time limit approaching ({elapsed:.2f}s), stopping refinement")
            break
'''
    
    # Replace lines[start_idx:end_idx+1] with new_section
    lines = lines[:start_idx] + new_section.splitlines(keepends=True) + lines[end_idx+1:]

with open('solver.py', 'w') as f:
    f.writelines(lines)
