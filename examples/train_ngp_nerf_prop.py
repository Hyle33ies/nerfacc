"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""
import argparse
import itertools
import pathlib
import time
import os
import json
from datetime import datetime
import shutil
from typing import Callable

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from lpips import LPIPS
from radiance_fields.ngp import NGPDensityField, NGPRadianceField

from utils import (
    MIPNERF360_UNBOUNDED_SCENES,
    NERF_SYNTHETIC_SCENES,
    render_image_with_propnet,
    set_random_seed,
)
from nerfacc.estimators.prop_net import (
    PropNetEstimator,
    get_proposal_requires_grad_fn,
)

def setup_logging(args):
    """Setup logging directory and files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs_ngp_prop", f"{args.scene}_{timestamp}")
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

def get_dir_size(path):
    """Calculate total size of a directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_root",
    type=str,
    default=str(pathlib.Path.cwd() / "../autodl-tmp/360_v2"),
    # default=str(pathlib.Path.cwd() / "../autodl-tmp/nerf_synthetic/nerf_synthetic"),
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
    "--scene",
    type=str,
    default="lego",
    choices=NERF_SYNTHETIC_SCENES + MIPNERF360_UNBOUNDED_SCENES,
    help="which scene to use",
)
parser.add_argument(
    "--test_chunk_size",
    type=int,
    default=8192,
)
args = parser.parse_args()

device = "cuda:0"
set_random_seed(42)

if args.scene in MIPNERF360_UNBOUNDED_SCENES:
    from datasets.nerf_360_v2 import SubjectLoader

    # training parameters
    max_steps = 12000
    init_batch_size = 4096
    weight_decay = 0.0
    # scene parameters
    unbounded = True
    aabb = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device=device)
    near_plane = 0.2  # TODO: Try 0.02
    far_plane = 1e3
    # dataset parameters
    train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
    test_dataset_kwargs = {"factor": 4}
    # model parameters
    proposal_networks = [
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=128,
        ).to(device),
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=256,
        ).to(device),
    ]
    # render parameters
    num_samples = 48
    num_samples_per_prop = [256, 96]
    sampling_type = "lindisp"
    opaque_bkgd = True

else:
    from datasets.nerf_synthetic import SubjectLoader

    # training parameters
    max_steps = 12000
    init_batch_size = 4096
    weight_decay = (
        1e-5 if args.scene in ["materials", "ficus", "drums"] else 1e-6
    )
    # scene parameters
    unbounded = False
    aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
    near_plane = 2.0
    far_plane = 6.0
    # dataset parameters
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}
    # model parameters
    proposal_networks = [
        NGPDensityField(
            aabb=aabb,
            unbounded=unbounded,
            n_levels=5,
            max_resolution=128,
        ).to(device),
    ]
    # render parameters
    num_samples = 64
    num_samples_per_prop = [128]
    sampling_type = "uniform"
    opaque_bkgd = False

train_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split=args.train_split,
    num_rays=init_batch_size,
    device=device,
    **train_dataset_kwargs,
)

test_dataset = SubjectLoader(
    subject_id=args.scene,
    root_fp=args.data_root,
    split="test",
    num_rays=None,
    device=device,
    **test_dataset_kwargs,
)

# setup the radiance field we want to train.
prop_optimizer = torch.optim.Adam(
    itertools.chain(
        *[p.parameters() for p in proposal_networks],
    ),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
prop_scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            prop_optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            prop_optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
estimator = PropNetEstimator(prop_optimizer, prop_scheduler).to(device)

grad_scaler = torch.cuda.amp.GradScaler(2**10)
radiance_field = NGPRadianceField(aabb=aabb, unbounded=unbounded).to(device)
optimizer = torch.optim.Adam(
    radiance_field.parameters(),
    lr=1e-2,
    eps=1e-15,
    weight_decay=weight_decay,
)
scheduler = torch.optim.lr_scheduler.ChainedScheduler(
    [
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=100
        ),
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[
                max_steps // 2,
                max_steps * 3 // 4,
                max_steps * 9 // 10,
            ],
            gamma=0.33,
        ),
    ]
)
proposal_requires_grad_fn = get_proposal_requires_grad_fn()
# proposal_annealing_fn = get_proposal_annealing_fn()

lpips_net = LPIPS(net="vgg").to(device)
lpips_norm_fn = lambda x: x[None, ...].permute(0, 3, 1, 2) * 2 - 1
lpips_fn = lambda x, y: lpips_net(lpips_norm_fn(x), lpips_norm_fn(y)).mean()

# Setup logging
log_dir = setup_logging(args)

# training
tic = time.time()
for step in range(max_steps + 1):
    radiance_field.train()
    for p in proposal_networks:
        p.train()
    estimator.train()

    i = torch.randint(0, len(train_dataset), (1,)).item()
    data = train_dataset[i]

    render_bkgd = data["color_bkgd"]
    rays = data["rays"]
    pixels = data["pixels"]

    proposal_requires_grad = proposal_requires_grad_fn(step)
    # render
    rgb, acc, depth, extras = render_image_with_propnet(
        radiance_field,
        proposal_networks,
        estimator,
        rays,
        # rendering options
        num_samples=num_samples,
        num_samples_per_prop=num_samples_per_prop,
        near_plane=near_plane,
        far_plane=far_plane,
        sampling_type=sampling_type,
        opaque_bkgd=opaque_bkgd,
        render_bkgd=render_bkgd,
        # train options
        proposal_requires_grad=proposal_requires_grad,
    )
    estimator.update_every_n_steps(
        extras["trans"], proposal_requires_grad, loss_scaler=1024
    )

    # compute loss
    loss = F.smooth_l1_loss(rgb, pixels)

    optimizer.zero_grad()
    # do not unscale it because we are using Adam.
    grad_scaler.scale(loss).backward()
    optimizer.step()
    scheduler.step()

    if step % 2000 == 0:
        elapsed_time = time.time() - tic
        loss = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(loss) / np.log(10.0)
        metrics = {
            "step": step,
            "elapsed_time": f"{elapsed_time:.2f}",
            "loss": f"{loss:.5f}",
            "psnr": f"{psnr:.2f}",
            "num_rays": len(pixels),
            "max_depth": f"{depth.max():.3f}"
        }
        print(
            f"elapsed_time={elapsed_time:.2f}s | step={step} | "
            f"loss={loss:.5f} | psnr={psnr:.2f} | "
            f"num_rays={len(pixels):d} | "
            f"max_depth={depth.max():.3f} | "
        )
        save_training_log(log_dir, step, metrics)
        
        # Save training visualization
        if step % 4000 == 0:
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
        # evaluation
        radiance_field.eval()
        for p in proposal_networks:
            p.eval()
        estimator.eval()

        psnrs = []
        lpips = []
        fps_list = []
        test_results = []
        with torch.no_grad():
            for i in tqdm.tqdm(range(len(test_dataset))):
                data = test_dataset[i]
                render_bkgd = data["color_bkgd"]
                rays = data["rays"]
                pixels = data["pixels"]

                # Measure rendering time
                start_time = time.time()
                # rendering
                (rgb, acc, depth, _,) = render_image_with_propnet(
                    radiance_field,
                    proposal_networks,
                    estimator,
                    rays,
                    # rendering options
                    num_samples=num_samples,
                    num_samples_per_prop=num_samples_per_prop,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    sampling_type=sampling_type,
                    opaque_bkgd=opaque_bkgd,
                    render_bkgd=render_bkgd,
                    # test options
                    test_chunk_size=args.test_chunk_size,
                )
                render_time = time.time() - start_time
                fps = 1.0 / render_time

                mse = F.mse_loss(rgb, pixels)
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                lpips_score = lpips_fn(rgb, pixels).item()
                
                # Save results for each test image
                test_results.append({
                    "image_id": i,
                    "psnr": psnr.item(),
                    "lpips": lpips_score,
                    "fps": fps,
                    "mse": mse.item()
                })
                psnrs.append(psnr.item())
                lpips.append(lpips_score)
                fps_list.append(fps)

            # Find best and worst cases
            best_idx = max(range(len(test_results)), key=lambda i: test_results[i]["psnr"])
            worst_idx = min(range(len(test_results)), key=lambda i: test_results[i]["psnr"])
            
            # Save visualization for best and worst cases
            vis_dir = os.path.join(log_dir, "test_vis")
            os.makedirs(vis_dir, exist_ok=True)
            
            for case, idx in [("best", best_idx), ("worst", worst_idx)]:
                data = test_dataset[idx]
                rgb, acc, depth, _ = render_image_with_propnet(
                    radiance_field,
                    proposal_networks,
                    estimator,
                    data["rays"],
                    # rendering options
                    num_samples=num_samples,
                    num_samples_per_prop=num_samples_per_prop,
                    near_plane=near_plane,
                    far_plane=far_plane,
                    sampling_type=sampling_type,
                    opaque_bkgd=opaque_bkgd,
                    render_bkgd=data["color_bkgd"],
                    # test options
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
        fps_avg = sum(fps_list) / len(fps_list)
        
        # Calculate storage space
        storage_size = get_dir_size(log_dir)
        
        # Save final test results
        final_results = {
            "average_metrics": {
                "psnr": psnr_avg,
                "lpips": lpips_avg,
                "fps": fps_avg
            },
            "storage_size_bytes": storage_size,
            "storage_size_mb": storage_size / (1024 * 1024),
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
        
        print(f"evaluation: psnr_avg={psnr_avg}, lpips_avg={lpips_avg}, fps_avg={fps_avg:.2f}")
        print(f"storage size: {storage_size / (1024 * 1024):.2f} MB")
        print(f"Results saved to {log_dir}")
