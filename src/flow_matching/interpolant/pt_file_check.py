import torch

# Load the file
data = torch.load("chromatin_data.pt")

# 1. Check Metadata
metadata = data.get("metadata", {})
print("--- Metadata ---")
for key, value in metadata.items():
    print(f"{key}: {value}")

# 2. Check the Actual Tensors
trajectories = data["trajectories"]
num_traj = len(trajectories)
print(f"\n--- Tensor Verification ---")
print(f"Number of trajectories in list: {num_traj}")

# Inspect the first trajectory
first_traj = trajectories[0]
print(f"Shape of first trajectory: {first_traj.shape}  # (M, K, 3)")
print(f"Data type: {first_traj.dtype}")

# 3. Check for consistency across all trajectories
m_shapes = [t.shape[0] for t in trajectories]
k_shapes = [t.shape[1] for t in trajectories]

if len(set(m_shapes)) == 1 and len(set(k_shapes)) == 1:
    print(f"All trajectories have consistent shapes: M={m_shapes[0]}, K={k_shapes[0]}")
else:
    print("Warning: Inconsistent trajectory lengths or atom counts detected!")