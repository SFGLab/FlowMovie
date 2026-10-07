import os
import torch
from torch.utils.data import DataLoader
import argparse

from src.flow_matching.flow_matching.FM_model import load_FM_model
from src.flow_matching.interpolant.interpolant_DataLoader import TrajectoryDataset
from src.md.visualization_utils import visualize_whole_genome, create_gif


@torch.no_grad()
def iterate_trajectory_pytorch(model, x0, num_steps=100, save_every=10, method="rk4", end_time=1.0):
    """
    x0: torch.Tensor of shape (3m,) or (m, 3)
    """
    device = x0.device if isinstance(x0, torch.Tensor) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not isinstance(x0, torch.Tensor):
        x0 = torch.tensor(x0, dtype=torch.float32, device=device)
    
    t_span = torch.linspace(0.0, end_time, num_steps, device=device)
    dt = t_span[1] - t_span[0]
    
    x_curr = x0.clone()
    structures = [x_curr.cpu().numpy()]
    
    model.eval()
    for i in range(len(t_span) - 1):
        t = t_span[i]

        if method == "euler":
            # Simple Forward Euler: x_next = x + v * dt
            v = model(x_curr, t.unsqueeze(0))
            x_curr = x_curr + v * dt
            
        elif method == "rk4":
            # 4th Order Runge-Kutta
            k1 = model(x_curr, t.unsqueeze(0))
            k2 = model(x_curr + 0.5 * dt * k1, (t + 0.5 * dt).unsqueeze(0))
            k3 = model(x_curr + 0.5 * dt * k2, (t + 0.5 * dt).unsqueeze(0))
            k4 = model(x_curr + dt * k3, (t + dt).unsqueeze(0))
            
            x_curr = x_curr + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        if (i + 1) % save_every == 0:
            structures.append(x_curr.cpu().numpy())
            
    return structures


def main(test_name, phase_trans, resolution=2_000_000, p=0, duration=300):
    """
    Based on the FM model from test_name and phase_trans, creates a vizualization gif.
    resolution, p and duration are the gif parameters.
    Parameter p signifies the percentage (integer from 0 to 100) of the points to exclude due to them lying to close to the center. 
    By default p=0, meaning that all of the points will be shown.
    """
    # Load pretrained model:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f"../../../models/FM/{test_name}_FM_model.pt"
    FM_model = load_FM_model(model_path, device)
    FM_model.eval()

    # Load the data:
    pt_file_path = f"../../../data/md/{test_name}/processed/{phase_trans}_data.pt"
    dataset = TrajectoryDataset(pt_file_path)
    full_batch_size = len(dataset)
    dataloader = DataLoader(dataset, batch_size=full_batch_size, shuffle=True)
    batch_trajectories = next(iter(dataloader)).to(device)
    # batch_trajectories shape: (Total_Dataset_Size, N, D)
    B, N, D = batch_trajectories.shape
    data_x0 = batch_trajectories[:, 0, :]   # First frame (B, D)

    # Iterating the trajectory:
    structures = iterate_trajectory_pytorch(FM_model, data_x0)
    structures = [struct[0,:-2] for struct in structures]
    structures = [struct.reshape((int(struct.shape[0]/3), 3)) for struct in structures]

    # Visualization:
    viz_path = f"../../../results/FM/{test_name}/viz"
    if not os.path.exists(viz_path):
        os.makedirs(viz_path)

    x0_3d = structures[0]
    axis_ranges = [x0_3d[:,0].min(), x0_3d[:,0].max(),
                x0_3d[:,1].min(), x0_3d[:,1].max(),
                x0_3d[:,2].min(), x0_3d[:,2].max()]
    figs = [visualize_whole_genome(
            points, 
            diploid=True, 
            chrom_sizes_path="../../../chrom_sizes/mm10.txt",
            resolution=resolution, p=p, title=f"Frame {i}", 
            fig_show=False, axis_range=axis_ranges) \
        for i, points in enumerate(structures)]

    path_gif = os.path.join(viz_path, "traj_FM.gif")
    create_gif(figs, path_gif, duration=duration)
    print(f"Successfully created gif at {os.path.join(viz_path, 'traj_FM.gif')}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pipeline with specified resolution and duration settings."
    )
    parser.add_argument(
        "--test-name",
        "-t",
        type=str,
        default="test01",
        help="Name of the test directory (default: 'test01')",
    )
    parser.add_argument(
        "--phase-trans",
        "-p",
        type=str,
        default="G1_to_S",
        help="Phase transition identifier (default: 'G1_to_S')",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=int,
        default=2_000_000,
        help="Resolution value (default: 2000000)",
    )
    parser.add_argument(
        "--percent-hidden",
        "--p",
        dest="p",
        type=int,
        default=0,
        help="Percent of the beads not to be visualized (value from 0 to 100, default: 0)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=300,
        help="Duration parameter (default: 300)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(args.test_name)
    main(
        args.test_name,
        args.phase_trans,
        resolution=args.resolution,
        p=args.p,
        duration=args.duration,
    )

