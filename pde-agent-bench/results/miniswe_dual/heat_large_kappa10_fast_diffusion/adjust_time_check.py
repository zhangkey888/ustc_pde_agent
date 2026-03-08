with open('solver.py', 'r') as f:
    content = f.read()

# Replace the time check in the loop
old_check = '''        # Check if we have time for this configuration
        elapsed = time.time() - start_time
        if elapsed > 0.8 * time_limit:  # Leave 20% margin for evaluation
            if rank == 0:
                print(f"  Time limit approaching ({elapsed:.2f}s), stopping refinement")
            break'''

new_check = '''        # Check if we have time for this configuration
        elapsed = time.time() - start_time
        if elapsed > 0.7 * time_limit:  # Leave 30% margin for evaluation
            if rank == 0:
                print(f"  Time limit approaching ({elapsed:.2f}s), stopping refinement")
            break'''

content = content.replace(old_check, new_check)

with open('solver.py', 'w') as f:
    f.write(content)
