import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
import torchvision.transforms as transforms
import time
import json
from timm.utils import unwrap_model

# Import the model architectures
from src.zoo.image import model_architectures as architectures

# Add deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define a dataset class for Vimeo-90k that returns frames to be compressed
class Vimeo90kFramesDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, septuplet_list, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            septuplet_list (string): Path to the file with list of septuplets.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.septuplet_list = []

        with open(septuplet_list, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    self.septuplet_list.append(line.strip())

    def __len__(self):
        return len(self.septuplet_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        septuplet_name = self.septuplet_list[idx]
        # We need to return the raw frames and the septuplet name for saving
        frames = []

        # Load frames
        for i in range(1, 7):  # Only frames 1-6 need to be compressed as reference for P frames
            img_name = os.path.join(self.root_dir, septuplet_name, f'im{i}.png')
            image = Image.open(img_name).convert('RGB')
            frames.append(image)

        # Apply transform if provided
        if self.transform:
            frames = [self.transform(img) for img in frames]

        return torch.stack(frames), septuplet_name  # Return frames as tensor [6, C, H, W] and path

def precompute_reference_frames(dataset, i_frame_model, device, save_dir, quality_index, batch_size=16, num_workers=4):
    """
    Precompute compressed reference frames using I-frame model
    
    Args:
        dataset: Dataset containing original frames
        i_frame_model: Pretrained I-frame model
        device: Device to run I-frame model on
        save_dir: Base directory to save compressed frames
        quality_index: Quality index (0-3) to organize the saved frames
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
    """
    # Create dataloader for efficient processing
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Set model to eval mode
    i_frame_model.eval()
    
    # Create the quality-specific save directory
    quality_save_dir = os.path.join(save_dir, str(quality_index))
    os.makedirs(quality_save_dir, exist_ok=True)
    
    # Create metadata to store precomputed frame info
    metadata = {
        'frame_count': 0,
        'quality_index': quality_index,
        'paths': {}
    }
    
    # Process all frames
    with torch.no_grad():
        for batch_frames, batch_paths in tqdm(dataloader, desc=f"Compressing frames with I-frame model (Quality {quality_index})"):
            # Process each frame in the batch
            batch_size, num_frames, C, H, W = batch_frames.shape
            
            # Reshape for batch processing: [B*6, C, H, W]
            reshaped_frames = batch_frames.view(-1, C, H, W).to(device)
            
            # Process in smaller sub-batches if needed (to avoid OOM)
            sub_batch_size = 1024  # Adjust based on your GPU memory
            compressed_frames = []
            
            for i in range(0, reshaped_frames.size(0), sub_batch_size):
                sub_batch = reshaped_frames[i:i+sub_batch_size]
                # Compress frames using I-frame model
                results = i_frame_model(sub_batch)
                compressed_frames.append(results["x_hat"].cpu())
            
            # Concatenate all sub-batches
            all_compressed = torch.cat(compressed_frames, dim=0)
            
            # Reshape back to [B, 6, C, H, W]
            all_compressed = all_compressed.view(batch_size, num_frames, C, H, W)
            
            # Save each sequence's compressed frames
            for b in range(batch_size):
                septuplet_path = batch_paths[b]
                
                # Create destination directory with quality index
                # Format: /save_dir/quality_index/xxxxx/xxxx/
                frame_save_dir = os.path.join(quality_save_dir, septuplet_path)
                os.makedirs(frame_save_dir, exist_ok=True)
                
                # Save each compressed frame in the sequence
                for f in range(num_frames):
                    frame_tensor = all_compressed[b, f]
                    frame_tensor = frame_tensor.clamp(0, 1)

                    # Save as compressed image
                    frame_path = os.path.join(frame_save_dir, f"ref{f+1}.png")
                    # Convert tensor to PIL image and as uncompressed PNG
                    frame_image = transforms.ToPILImage()(frame_tensor)
                    frame_image.save(frame_path, format='PNG', compress_level=0)
                
                # Update metadata
                metadata['frame_count'] += num_frames
                metadata['paths'][septuplet_path] = frame_save_dir
    
    # Save metadata for this quality level
    with open(os.path.join(quality_save_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    print(f"Precomputed {metadata['frame_count']} reference frames for quality {quality_index}, saved to {quality_save_dir}")
    return metadata

def main():
    parser = argparse.ArgumentParser(description='Precompute Reference Frames with I-frame Model')
    parser.add_argument('--vimeo_dir', type=str, default='/Vimeo90k/vimeo_septuplet/sequences', 
                        help='Path to Vimeo-90k dataset')
    parser.add_argument('--save_dir', type=str, default='/Vimeo90k/vimeo_septuplet/reference_sequences', 
                        help='Base directory to save precomputed reference frames')
    parser.add_argument('--septuplet_list', type=str, required=True, help='Path to septuplet list file')
    parser.add_argument('--i_frame_model_name', type=str, default='cheng2020-anchor', help='I-frame model name')
    parser.add_argument('--i_frame_model_path', type=str, required=True, help='Path to I-frame model checkpoint')
    parser.add_argument('--quality_index', type=int, required=True, choices=[0, 1, 2, 3], help='Quality index (0-3)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')
    parser.add_argument('--cuda_device', type=str, default='0', help='CUDA device indices')
    parser.add_argument('--num_workers', type=int, default=64, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--compile', action='store_true', help='Compile the model')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set CUDA devices
    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create dataset for frames to compress
    dataset = Vimeo90kFramesDataset(
        root_dir=args.vimeo_dir,
        septuplet_list=args.septuplet_list,
        transform=transform
    )

    print(f"Dataset loaded with {len(dataset)} sequences.")

    # Load I-frame model
    i_frame_load_checkpoint = torch.load(args.i_frame_model_path, map_location=torch.device('cpu'))
    i_frame_model = architectures[args.i_frame_model_name].from_state_dict(i_frame_load_checkpoint).eval()
    i_frame_model = i_frame_model.to(device)
    
    # Compile model if specified
    if args.compile:
        print("Compiling the I-frame model...")
        i_frame_model = torch.compile(i_frame_model)

    # Precompute and save reference frames
    start_time = time.time()
    metadata = precompute_reference_frames(
        dataset=dataset,
        i_frame_model=i_frame_model,
        device=device,
        save_dir=args.save_dir,
        quality_index=args.quality_index,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    elapsed_time = time.time() - start_time
    
    print(f"Precomputation completed in {elapsed_time:.2f} seconds")
    print(f"Processed {metadata['frame_count']} frames from {len(metadata['paths'])} sequences")
    print(f"Results saved to {args.save_dir}")

if __name__ == '__main__':
    main()