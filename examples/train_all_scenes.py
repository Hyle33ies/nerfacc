import subprocess
import time
import os
# from utils import NERF_SYNTHETIC_SCENES
from utils import MIPNERF360_UNBOUNDED_SCENES

def run_training(scene, method, train_split="train"):
    """Run training for a single scene with specified method"""
    print(f"\n{'='*50}")
    print(f"Training scene: {scene} with {method}")
    print(f"{'='*50}\n")
    
    # Select the appropriate training script based on method
    script = "train_ngp_nerf_occ.py" if method == "occ" else "train_ngp_nerf_prop.py"
    
    cmd = [
        "python", f"examples/{script}",
        "--scene", scene,
        "--train_split", train_split,
        "--data_root", "../autodl-tmp/360_v2"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nSuccessfully completed {method} training for {scene}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError in {method} training for {scene}: {e}")
        return False

def main():
    # List of scenes to train
    scenes = MIPNERF360_UNBOUNDED_SCENES
    methods = ["occ", "prop"]  # Training methods to use
    
    # Create a log file to track progress
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"training_progress_{timestamp}.txt"
    
    with open(log_file, "w") as f:
        f.write("Training Progress Log\n")
        f.write("=" * 50 + "\n\n")
    
    # Train each scene with each method
    for scene in scenes:
        for method in methods:
            start_time = time.time()
            
            # Run training
            success = run_training(scene, method)
            
            # Calculate time taken
            time_taken = time.time() - start_time
            
            # Log the result
            with open(log_file, "a") as f:
                status = "SUCCESS" if success else "FAILED"
                f.write(f"Scene: {scene}\n")
                f.write(f"Method: {method}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Time taken: {time_taken:.2f} seconds\n")
                f.write("-" * 50 + "\n")
            
            # Add a small delay between runs to ensure clean GPU memory
            time.sleep(5)
    
    print("\nTraining completed for all scenes and methods!")
    print(f"Check {log_file} for detailed results.")

if __name__ == "__main__":
    main() 
