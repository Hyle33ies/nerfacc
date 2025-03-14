"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import pathlib
import time
import os
import json
from datetime import datetime

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets.nerf_synthetic import SubjectLoader
from lpips import LPIPS
from radiance_fields.mlp import VanillaNeRFRadianceField
from torch.optim.adam import Adam
from utils import (
    NERF_SYNTHETIC_SCENES,
    render_image_with_occgrid,
    set_random_seed,
)
from nerfacc.estimators.occ_grid import OccGridEstimator

def setup_logging(args):
    """Setup logging directory and files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs_mlp", f"{args.scene}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    # Save training config
    config = vars(args)
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    return log_dir

def save_training_log(log_dir, step, metrics):
    """Save training metrics to log file"""
    log_file = os.path.join(log_dir, "training_log.txt")
    with open(log_file, "a") as f:
        f.write(f"{metrics}\n")

def save_test_results(log_dir, results):
    """Save test results and sample images"""
    # Save metrics
    with open(os.path.join(log_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)

device = torch.device("cuda:0")
set_random_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str(pathlib.Path.cwd() / "../autodl-tmp/nerf_synthetic/nerf_synthetic"),
    help="the root dir of the dataset",
)
parser.add_argument(
    "--train_split",
    type=str,
    default="train",
    choices=["train", "trainval"],
    help="which train split to use",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None,
    help="the path of the pretrained model",
)
parser.add_argument(
    "--scene",
    type=str,
    default="lego",
    choices=NERF_SYNTHETIC_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=4096,
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="results",
    help="directory to save results",
)
args = parser.parse_args()

# Setup logging
log_dir = setup_logging(args)

# training parameters
max_steps = 50000
init_batch_size = 1024
target_sample_batch_size = 1 << 16
# scene parameters
aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
near_plane = 0.0
far_plane = 1.0e10
# model parameters
grid_resolution = 128
grid_nlvl = 1
# render parameters
render_step_size = 5e-3

# setup the dataset
train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
)
test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
)

estimator = OccGridEstimator(
    roi_aabb=aabb, resolution=grid_resolution, levels=grid_nlvl
).to(device)

# setup the radiance field we want to train.
radiance_field = VanillaNeRFRadianceField().to(device)
optimizer = Adam(radiance_field.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[
        max_steps // 2,
        max_steps * 3 // 4,
        max_steps * 5 // 6,
        max_steps * 9 // 10,
    ],
    gamma=0.33,
)

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

if args.model_path is not None:
    checkpoint = torch.load(args.model_path)
    radiance_field.load_state_dict(checkpoint["radiance_field_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    estimator.load_state_dict(checkpoint["estimator_state_dict"])
    step = checkpoint["step"]
else:
    step = 0

# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    def occ_eval_fn(x):
        density = radiance_field.query_density(x)
        return density * render_step_size

    # update occupancy grid
    estimator.update_every_n_steps(
        step=step,
        occ_eval_fn=occ_eval_fn,
        occ_thre=1e-2,
    )

    # render
    rgb, acc, depth, n_rendering_samples = render_image_with_occgrid(
        radiance_field,
        estimator,
        rays,
        near_plane=near_plane,
        render_step_size=render_step_size,
        render_bkgd=render_bkgd,
    )
    if n_rendering_samples == 0:
        continue

    if target_sample_batch_size > 0:
        # Adjust number of rays to maintain constant sample batch size
        num_rays = len(pixels)
        num_rays = int(
            num_rays * (target_sample_batch_size / float(n_rendering_samples))
        )
        train_dataset.update_num_rays(num_rays)

    # compute loss
    loss = F.smooth_l1_loss(rgb, pixels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if step % 5000 == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        metrics = {
            "step": step,
            "elapsed_time": f"{elapsed_time:.2f}",
            "loss": f"{loss:.5f}",
            "psnr": f"{psnr:.2f}",
            "n_rendering_samples": n_rendering_samples,
            "num_rays": len(pixels),
            "max_depth": f"{depth.max():.3f}"
        }
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"n_rendering_samples={n_rendering_samples:d} | num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )
        save_training_log(log_dir, step, metrics)
        
        # Save training visualization
        if step % 10000 == 0:
            vis_dir = os.path.join(log_dir, "train_vis")
            os.makedirs(vis_dir, exist_ok=True)
            imageio.imwrite(
                os.path.join(vis_dir, f"rgb_step_{step}.png"),
                (rgb.detach().cpu().numpy() * 255).astype(np.uint8),
            )
            imageio.imwrite(
                os.path.join(vis_dir, f"depth_step_{step}.png"),
                np.repeat((depth.detach().cpu().numpy() * 255 / depth.max().item()).astype(np.uint8)[..., np.newaxis], 3, axis=-1).squeeze(),
            )

    if step > 0 and step % max_steps == 0:
        # Save model checkpoint
        model_save_path = os.path.join(log_dir, f"checkpoint_{step}.pt")
        torch.save(
            {
                "step": step,
                "radiance_field_state_dict": radiance_field.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "estimator_state_dict": estimator.state_dict(),
            },
            model_save_path,
        )
        print(f"Saved checkpoint to {model_save_path}")

        # evaluation
        radiance_field.eval()
        estimator.eval()

        psnrs = []
        lpips = []
        test_results = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # rendering
                rgb, acc, depth, _ = render_image_with_occgrid(
                    radiance_field,
                    estimator,
                    rays,
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=render_bkgd,
                    test_chunk_size=args.test_chunk_size,
                )
                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                lpips_score = lpips_fn(rgb, pixels).item()
                
                # Save results for each test image
                test_results.append({
                    "image_id": i,
                    "psnr": psnr.item(),
                    "lpips": lpips_score,
                    "mse": mse.item()
                })
                psnrs.append(psnr.item())
                lpips.append(lpips_score)

            # Find best and worst cases
            best_idx = max(range(len(test_results)), key=lambda i: test_results[i]["psnr"])
            worst_idx = min(range(len(test_results)), key=lambda i: test_results[i]["psnr"])
            
            # Save visualization for best and worst cases
            vis_dir = os.path.join(log_dir, "test_vis")
            os.makedirs(vis_dir, exist_ok=True)
            
            for case, idx in [("best", best_idx), ("worst", worst_idx)]:
                data = test_dataset[idx]
                rgb, acc, depth, _ = render_image_with_occgrid(
                    radiance_field,
                    estimator,
                    data["rays"],
                    near_plane=near_plane,
                    render_step_size=render_step_size,
                    render_bkgd=data["color_bkgd"],
                    test_chunk_size=args.test_chunk_size,
                )
                
                # Save rendered image, ground truth, and error map
                imageio.imwrite(
                    os.path.join(vis_dir, f"{case}_rendered.png"),
                    (rgb.cpu().numpy() * 255).astype(np.uint8),
                )
                imageio.imwrite(
                    os.path.join(vis_dir, f"{case}_ground_truth.png"),
                    (data["pixels"].cpu().numpy() * 255).astype(np.uint8),
                )
                imageio.imwrite(
                    os.path.join(vis_dir, f"{case}_error.png"),
                    ((rgb - data["pixels"]).norm(dim=-1).cpu().numpy() * 255).astype(np.uint8),
                )
                imageio.imwrite(
                    os.path.join(vis_dir, f"{case}_depth.png"),
                    np.repeat((depth.cpu().numpy() * 255 / depth.max().item()).astype(np.uint8)[..., np.newaxis], 3, axis=-1).squeeze(),
                )

        psnr_avg = sum(psnrs) / len(psnrs)
        lpips_avg = sum(lpips) / len(lpips)
        
        # Save final test results
        final_results = {
            "average_metrics": {
                "psnr": psnr_avg,
                "lpips": lpips_avg
            },
            "best_case": {
                "image_id": best_idx,
                **test_results[best_idx]
            },
            "worst_case": {
                "image_id": worst_idx,
                **test_results[worst_idx]
            },
            "all_results": test_results
        }
        save_test_results(log_dir, final_results)
        
        print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}")
        print(f"Results saved to {log_dir}")
