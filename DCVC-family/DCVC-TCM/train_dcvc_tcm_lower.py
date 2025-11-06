import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import random
import math
from PIL import Image
import torchvision.transforms as transforms
import time
from timm.utils import unwrap_model
from torch_ema import ExponentialMovingAverage

# Import the model
from src.models.video_net_dmc import DMC
from src.models.priors import model_architectures as architectures


# Add deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Define a dataset class for Vimeo-90k that returns GOP sequences
class Vimeo90kGOPDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, precomputed_dir, septuplet_list, transform=None, crop_size=256, gop_size=7, shuffle_frames=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            septuplet_list (string): Path to the file with list of septuplets.
            transform (callable, optional): Optional transform to be applied on a sample.
            crop_size (int): Size of the random crop.
            gop_size (int): GOP size for training.
        """
        self.root_dir = root_dir
        self.precomputed_dir = precomputed_dir
        self.transform = transform
        self.crop_size = crop_size
        self.gop_size = gop_size
        self.shuffle_frames = shuffle_frames
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
        frames = []
        precomputed_frames = []

        # Load frames
        for i in range(2, 8):  # Vimeo-90k septuplet has 7 frames Frames 2-7 (to be compressed as P-frames)
            img_name = os.path.join(self.root_dir, septuplet_name, f'im{i}.png')
            image = Image.open(img_name).convert('RGB')
            frames.append(image)

        # Load precomputed frames
        for i in range(1, 7):  # Vimeo-90k septuplet has 7 frames Reference frames 1-6
            precomputed_name = os.path.join(self.precomputed_dir, septuplet_name, f'ref{i}.png')
            precomputed_image = Image.open(precomputed_name).convert('RGB')
            precomputed_frames.append(precomputed_image)

        # Apply random crop to the same location for all frames
        if self.crop_size:
            width, height = frames[0].size
            if width >= self.crop_size and height >= self.crop_size:
                x = random.randint(0, width - self.crop_size)
                y = random.randint(0, height - self.crop_size)
                frames = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in frames]
                precomputed_frames = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in precomputed_frames]

        if random.random() < 0.5:  # 50% chance to flip horizontally
            frames = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in frames]
            precomputed_frames = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in precomputed_frames]

        if random.random() < 0.5:  # 50% chance to flip vertically
            frames = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in frames]
            precomputed_frames = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in precomputed_frames]


        # Apply transform if provided
        if self.transform:
            frames = [self.transform(img) for img in frames]
            precomputed_frames = [self.transform(img) for img in precomputed_frames]


        # Random shuffle frame order if enabled
        if self.shuffle_frames:
            # Create a list of indices and shuffle it
            frame_indices = list(range(len(frames)))
            random.shuffle(frame_indices)
            
            # Reorder frames according to shuffled indices
            frames = [frames[i] for i in frame_indices]
            precomputed_frames = [precomputed_frames[i] for i in frame_indices]
            
            # Option: You could also return the shuffle indices if needed for reconstruction
            # return torch.stack(frames), frame_indices

        return torch.stack(frames), torch.stack(precomputed_frames)  # Return frames as a single tensor [S, C, H, W]

# Define a dataset class for UVG that returns GOP sequences
class UVGGOPDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, gop_size=12):
        """
        Args:
            root_dir (string): Directory containing UVG video frames.
            transform (callable, optional): Optional transform to be applied on a sample.
            gop_size (int): GOP size for training.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.gop_size = gop_size
        self.video_sequences = []

        # UVG videos
        video_names = [
            'Beauty_1920x1024_120fps_420_8bit_YUV', 'Bosphorus_1920x1024_120fps_420_8bit_YUV', 
            'HoneyBee_1920x1024_120fps_420_8bit_YUV', 'Jockey_1920x1024_120fps_420_8bit_YUV', 
            'ReadySteadyGo_1920x1024_120fps_420_8bit_YUV', 'ShakeNDry_1920x1024_120fps_420_8bit_YUV', 
            'YachtRide_1920x1024_120fps_420_8bit_YUV'
        ]

        # Get sequences of frames for each video
        for video_name in video_names:
            video_dir = os.path.join(root_dir, video_name)
            if os.path.isdir(video_dir):
                frames = sorted([f for f in os.listdir(video_dir) if f.endswith('.png') or f.endswith('.jpg')])

                # Divide frames into sequences of length gop_size (or maximum available)
                for i in range(0, len(frames), gop_size):
                    seq_frames = frames[i:min(i + gop_size, len(frames))]
                    if len(seq_frames) >= 2:  # Need at least 2 frames for P-frame training
                        self.video_sequences.append({
                            'video': video_name,
                            'frames': seq_frames
                        })

    def __len__(self):
        return len(self.video_sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sequence = self.video_sequences[idx]
        video_name = sequence['video']
        frame_names = sequence['frames']

        frames = []
        for frame_name in frame_names:
            frame_path = os.path.join(self.root_dir, video_name, frame_name)
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)

        # Apply transform if provided
        if self.transform:
            frames = [self.transform(img) for img in frames]

        return torch.stack(frames)  # Return frames as a single tensor [S, C, H, W]


def train_one_epoch_fully_batched(model, i_frame_model, train_loader, optimizer, device, stage, epoch, gradient_accumulation_steps=1, finetune=False, grad_clip_max_norm=None, ema=None,args=None):
    """
    Train for one epoch with fully batched processing for GOP sequences.
    
    Args:
        gop_size: Group of Pictures size (7 for Vimeo90k)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        grad_clip_max_norm: If not None, clip gradients to this max norm
        ema: Optional ExponentialMovingAverage for parameter updates
    """
    model.train()
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_psnr = 0
    n_frames = 0
    
    # Control parameter freezing based on stage - only do this once per epoch
    if stage in [2, 3]:  # Freeze MV generation part in stages 2 and 3
        for param in model.optic_flow.parameters():
            param.requires_grad = False
        for param in model.mv_encoder.parameters():
            param.requires_grad = False
        for param in model.mv_decoder.parameters():
            param.requires_grad = False
        for param in model.mv_prior_encoder.parameters():
            param.requires_grad = False
        for param in model.mv_prior_decoder.parameters():
            param.requires_grad = False
        for param in model.bit_estimator_z_mv.parameters():
            param.requires_grad = False
    else:  # Unfreeze MV generation part in stages 1 and 4
        for param in model.optic_flow.parameters():
            param.requires_grad = True
        for param in model.mv_encoder.parameters():
            param.requires_grad = True
        for param in model.mv_decoder.parameters():
            param.requires_grad = True
        for param in model.mv_prior_encoder.parameters():
            param.requires_grad = True
        for param in model.mv_prior_decoder.parameters():
            param.requires_grad = True
        for param in model.bit_estimator_z_mv.parameters():
            param.requires_grad = True
    
    # Process batches of GOP sequences
    progress_bar = tqdm(train_loader)
    
    for batch_idx, (batch_frames, batch_precomputed_frames) in enumerate(progress_bar):
        batch_size = batch_frames.size(0)
        seq_length = batch_frames.size(1)  # Number of frames in sequence (S dimension)
        batch_loss = 0
        
        # Process each frame position sequentially within the batch
        # For each position, we process all sequences in parallel
        
        # Initialize reference frames for the batch
        
        if finetune and stage == 4:
            # Process each frame position in the sequence
            reference_frames = None
            reference_features = None
            optimizer.zero_grad()
            
            # Tracking metrics for the current batch
            batch_mse = 0
            batch_bpp = 0
            batch_psnr = 0
            
            for frame_pos in range(seq_length):
                # Get all frames at this position from all sequences in the batch
                current_frames = batch_frames[:, frame_pos, :, :, :].to(device)  # Shape: [B, C, H, W]
                
                if frame_pos == 0:  # First frame (I-frame) in each sequence
                    reference_frames = batch_precomputed_frames[:, frame_pos, :, :, :].to(device)  # Shape: [B, C, H, W]
                    result = model.forward_one_frame(current_frames, reference_frames, reference_features, training=True, stage=stage)
                    
                else:
                    result = model.forward_one_frame(current_frames, reference_frames, reference_features, training=True, stage=stage)
                
                reference_frames = result["recon_image"]
                if not args.frame_only:
                    reference_features = result["feature"] 
                # Direct accumulation of losses and indicators, no storage of results
                batch_loss += result["loss"]
                batch_mse += result["mse_loss"].item() * batch_size
                batch_bpp += result["bpp_train"].item() * batch_size
                batch_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size

                
            # Calculate average loss and back propagate
            batch_loss /= seq_length
            batch_loss /= gradient_accumulation_steps
            batch_loss.backward()
            
            # Apply gradient clipping if specified
            if grad_clip_max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)
                
            optimizer.step()
            
            # Update EMA parameters if EMA is enabled
            if ema is not None:
                ema.update(model.parameters())
                
            # Update of total indicators
            total_loss += batch_loss.item() * seq_length  # Adjust to maintain correct loss accounting
            total_mse += batch_mse
            total_bpp += batch_bpp
            total_psnr += batch_psnr
            n_frames += batch_size * seq_length

        else:
            # print("Three frame training")
            reference_features = None
            # Process each frame position in the sequence
            for frame_pos in range(seq_length):
                if frame_pos == seq_length - 1:
                    continue
                reference_frames = batch_precomputed_frames[:, frame_pos, :, :, :].to(device)  # Shape: [B, C, H, W]
                current_frames = batch_frames[:, frame_pos, :, :, :].to(device)  # Shape: [B, C, H, W]
                next_frames = batch_frames[:, frame_pos + 1, :, :, :].to(device)  # Shape: [B, C, H, W]
                # Zero gradients for each frame position
                optimizer.zero_grad()

                # Process all P-frames in the batch with their corresponding reference frames
                # DCVC model expects reference and current frames in the same batch size
                result = model.forward_one_frame(current_frames, reference_frames, reference_features, training=True, stage=stage)

                # Update reference frames for next position
                reference_frames = result["recon_image"]  # Shape: [B, C, H, W]
                if not args.frame_only:
                    reference_features = result["feature"]
                # Process next frame with the updated reference frames
                result_next = model.forward_one_frame(next_frames, reference_frames, reference_features, training=True, stage=stage)

                # Calculate loss 
                loss = (result["loss"] + result_next["loss"])/2
                loss.backward()

                reference_features = None
                # Apply gradient clipping if specified
                if grad_clip_max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

                # Apply optimizer step for each frame position
                optimizer.step()
                
                # Update EMA parameters if EMA is enabled
                if ema is not None:
                    ema.update(model.parameters())

                # Collect statistics
                batch_loss += result["loss"].item()
                total_mse += result["mse_loss"].item() * batch_size  # Account for all frames in batch
                total_bpp += result["bpp_train"].item() * batch_size
                total_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size

                n_frames += batch_size  # Count all frames in batch
            
            # Update total loss
            total_loss += batch_loss
            
        # Update progress bar - single update for both conditions
        if n_frames > 0:
            progress_bar.set_description(
                f"Epoch {epoch} Stage {stage} | "
                f"Loss: {total_loss / n_frames:.4f}, "
                f"MSE: {total_mse / n_frames:.6f}, "
                f"BPP: {total_bpp / n_frames:.4f}, "
                f"PSNR: {total_psnr / n_frames:.4f}"
            )
        
        # clear cache
        if finetune and stage == 4:
            del batch_frames, batch_precomputed_frames
            del batch_loss
            torch.cuda.empty_cache()
    
    # Calculate epoch statistics
    if n_frames > 0:
        avg_loss = total_loss / n_frames
        avg_mse = total_mse / n_frames
        avg_psnr = -10 * math.log10(avg_mse) if avg_mse > 0 else 100
        avg_bpp = total_bpp / n_frames
    
    else:
        avg_loss = 0
        avg_mse = 0
        avg_psnr = 0
        avg_bpp = 0

    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "psnr": avg_psnr,
        "bpp": avg_bpp,
    }


def evaluate_fully_batched(model, i_frame_model, test_loader, device, stage, finetune=False,args = None):
    """
    Evaluate model using fully batched processing for GOP sequences in finetune mode
    
    Args:
        gop_size: Group of Pictures size (12 for UVG)
    """
    model.eval()
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_psnr = 0
    n_frames = 0
    
    with torch.no_grad():
        for batch_frames in test_loader:
            batch_size = batch_frames.size(0)
            seq_length = batch_frames.size(1)  # Number of frames in sequence
            
            # Initialize reference frames
            reference_frames = None
            reference_features = None
            
            # Process each frame position in the sequence
            for frame_pos in range(seq_length):
                # Get all frames at this position from all sequences in the batch
                current_frames = batch_frames[:, frame_pos, :, :, :].to(device)  # Shape: [B, C, H, W]
                
                if frame_pos == 0:  # I-frames
                    # Process all I-frames in the batch together
                    i_frame_results = i_frame_model.encode_decode(current_frames)
                    reference_frames = i_frame_results["x_hat"]  # Shape: [B, C, H, W]
                else:  # P-frames
                    # Process all P-frames in the batch with their corresponding reference frames
                    result = model.forward_one_frame(current_frames, reference_frames, reference_features, training=False, stage=stage)
                    
                    # Update reference frames for next position
                    reference_frames = result["recon_image"]  # Shape: [B, C, H, W]
                    if not args.frame_only:
                        reference_features = result["feature"]  # Shape: [B, C, H, W]
                    
                    # Collect statistics
                    total_loss += result["loss"].item() * batch_size
                    total_mse += result["mse_loss"].item() * batch_size
                    total_bpp += result["bpp_train"].item() * batch_size
                    total_psnr += -10 * math.log10(result["mse_loss"].item()) * batch_size
                    
                    n_frames += batch_size
    
    # Calculate average statistics
    if n_frames > 0:
        avg_loss = total_loss / n_frames
        avg_mse = total_mse / n_frames
        avg_psnr = -10 * math.log10(avg_mse) if avg_mse > 0 else 100
        avg_bpp = total_bpp / n_frames
    else:
        avg_loss = 0
        avg_mse = 0
        avg_psnr = 0
        avg_bpp = 0
    
    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "psnr": avg_psnr,
        "bpp": avg_bpp
    }

def evaluate_fully_batched_three(model, i_frame_model, test_loader, device, stage,args):
    """
    Evaluate model using fully batched processing for GOP sequences
    
    Args:
        gop_size: Group of Pictures size (12 for UVG)
    """
    model.eval()
    total_loss = 0
    total_mse = 0
    total_bpp = 0
    total_psnr = 0
    n_frames = 0
    
    with torch.no_grad():
        for batch_frames in test_loader:
            batch_size = batch_frames.size(0)
            seq_length = batch_frames.size(1)  # Number of frames in sequence
            
            # Initialize reference frames
            reference_frames = None
            reference_features = None
            
            for frame_pos in range(seq_length):
                if frame_pos == 0:
                    continue
                else:
                    if frame_pos == seq_length-1:
                        continue

                    previous_frames = batch_frames[:, frame_pos - 1, :, :, :].to(device)
                    current_frames = batch_frames[:, frame_pos, :, :, :].to(device)
                    next_frames = batch_frames[:, frame_pos + 1, :, :, :].to(device)
                    #process previous frames by I-frame model
                    i_frame_results = i_frame_model.encode_decode(previous_frames)
                    reference_frames = i_frame_results["x_hat"]  # Shape: [B, C, H, W]
                    # Process all P-frames in the batch with their corresponding reference frames
                    IP_result = model.forward_one_frame(current_frames, reference_frames, reference_features, training=False, stage=stage)
                    IP_loss = IP_result["loss"]
                    if not args.frame_only:
                        reference_features = IP_result["feature"]
                    PP_result = model.forward_one_frame(next_frames,IP_result["pixel_rec"] if stage==1 else IP_result["recon_image"], reference_features,training=False, stage=stage)
                    PP_loss = PP_result["loss"]
                    reference_features = None
                    # Collect statistics
                    total_loss += (IP_loss + PP_loss).item() / 2 * batch_size
                    total_mse += (IP_result["mse_loss"].item() + PP_result["mse_loss"].item()) / 2 * batch_size
                    total_bpp += (IP_result["bpp_train"].item() + PP_result["bpp_train"].item()) / 2 * batch_size
                    total_psnr += (-10 * math.log10((IP_result["mse_loss"].item() + PP_result["mse_loss"].item()) / 2)) * batch_size
                    n_frames += batch_size
    
    # Calculate average statistics
    if n_frames > 0:
        avg_loss = total_loss / n_frames
        avg_mse = total_mse / n_frames
        avg_psnr = -10 * math.log10(avg_mse) if avg_mse > 0 else 100
        avg_bpp = total_bpp / n_frames
    else:
        avg_loss = 0
        avg_mse = 0
        avg_psnr = 0
        avg_bpp = 0
    
    return {
        "loss": avg_loss,
        "mse": avg_mse,
        "psnr": avg_psnr,
        "bpp": avg_bpp
    }

def main():
    parser = argparse.ArgumentParser(description='DCVC Training with Full Batch Processing')
    parser.add_argument('--vimeo_dir', type=str, required=True, help='Path to Vimeo-90k dataset')
    parser.add_argument('--precomputed_dir', type=str, required=True, help='Path to precomputed directory')
    parser.add_argument('--septuplet_list', type=str, required=True, help='Path to septuplet list file')
    parser.add_argument('--i_frame_model_name', type=str, default='IntraNoAR', help='I-frame model name')
    parser.add_argument('--i_frame_model_path', type=str, required=True, help='Path to I-frame model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='results/checkpoints_lower', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='results/logs_lower', help='Directory to save logs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--crop_size', type=int, default=256, help='Random crop size')
    parser.add_argument('--lambda_value', type=float, required=True, help='Lambda value for rate-distortion trade-off')
    parser.add_argument('--quality_index', type=int, required=True, help='Quality index (0-3) for the model name')
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3, 4], help='Training stage (1-4)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for this stage')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--cuda', type=bool, default=True, help='Use CUDA')
    parser.add_argument('--cuda_device', type=str, default='1', help='CUDA device indices')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--model_type', type=str, default='psnr', choices=['psnr', 'ms-ssim'],
                        help='Model type: psnr or ms-ssim')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--previous_stage_checkpoint', type=str, default=None,
                        help='Path to checkpoint from previous stage to resume from')
    parser.add_argument('--not_skip_test', action='store_true', help='Skip test during training')
    parser.add_argument('--uvg_dir', type=str, required=True, help='Path to UVG dataset')
    parser.add_argument('--frame_only', action='store_true', help='Use frame-only model')
    
    # Add arguments for resume training and SpyNet initialization
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--spynet_checkpoint', type=str, default=None, 
                       help='Path to SpyNet pretrained weights to initialize motion estimation network')
    parser.add_argument('--spynet_from_dcvc_checkpoint', type=str, default=None,
                        help='Path to DCVC checkpoint to initialize SpyNet weights')
    
    # Add learning rate scheduler arguments
    parser.add_argument('--lr_scheduler', type=str, default='step', 
                        choices=['step', 'multistep', 'cosine', 'plateau', 'none'],
                        help='Type of learning rate scheduler')
    parser.add_argument('--lr_step_size', type=int, default=5, 
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.5, 
                        help='Gamma for StepLR and MultiStepLR schedulers (multiplicative factor)')
    parser.add_argument('--lr_milestones', type=str, default='5,10,15', 
                        help='Comma-separated milestone epochs for MultiStepLR scheduler')
    parser.add_argument('--lr_min_factor', type=float, default=0.01, 
                        help='Minimum lr factor for CosineAnnealingLR scheduler (as a fraction of initial lr)')
    parser.add_argument('--lr_patience', type=int, default=2, 
                        help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--lr_warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs with linearly increasing LR')
    parser.add_argument('--reset_lr', action='store_true')
    
    # Add gradient accumulation option
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of steps to accumulate gradients (for larger effective batch sizes)')
    # torch.compile
    parser.add_argument('--compile', action='store_true', help='Compile the model')

    # finetune indicator
    parser.add_argument('--finetune', action='store_true', help='Finetune the model')
    parser.add_argument('--finetune_checkpoint', type=str, default=None,
                        help='Path to checkpoint to finetune from')
                        
    # Add gradient clipping parameter
    parser.add_argument('--grad_clip_max_norm', type=float, default=2.0,
                        help='Maximum norm for gradient clipping (None to disable clipping)')
                        
    # Add EMA arguments
    parser.add_argument('--use_ema', action='store_true', help='Use Exponential Moving Average for model weights')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate (higher = slower adaptation)')

    # Add arguments for global best tracking and evaluation
    parser.add_argument('--global_best_dir', type=str, default='best_ckpt_lower',
                    help='Directory to save global best checkpoints across stages')
    parser.add_argument('--evaluate_both', action='store_true', 
                    help='Evaluate both normal and EMA weights when EMA is enabled')

    args = parser.parse_args()

    if args.finetune:
        #set new dir as orig ckpt_dir_finetune
        args.checkpoint_dir = args.checkpoint_dir+'_finetune'
        args.log_dir = args.log_dir+'_finetune'
        args.global_best_dir = args.global_best_dir+'_finetune'

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

    # Create checkpoint and log directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    if args.global_best_dir:
        os.makedirs(args.global_best_dir, exist_ok=True)

    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    
    # Create training dataset with GOP structure
    train_dataset = Vimeo90kGOPDataset(
        root_dir=args.vimeo_dir,
        precomputed_dir=os.path.join(args.precomputed_dir, str(args.quality_index)),
        septuplet_list=args.septuplet_list,
        transform=transform,
        crop_size=args.crop_size,
        gop_size=7  # Vimeo90k has 7 frames per sequence
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Create test dataset (UVG) with GOP structure
    test_dataset = UVGGOPDataset(
        root_dir=args.uvg_dir,
        transform=transform,
        gop_size=12
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Make script print info about UVG dataset at start
    print(f"UVG dataset loaded with {len(test_dataset)} sequences.")

    # Load I-frame model
    i_frame_load_checkpoint = torch.load(args.i_frame_model_path, map_location=torch.device('cpu'))
    if "state_dict" in i_frame_load_checkpoint:
        i_frame_load_checkpoint = i_frame_load_checkpoint['state_dict']
    i_frame_model = architectures[args.i_frame_model_name].from_state_dict(i_frame_load_checkpoint).eval()
    i_frame_model = i_frame_model.to(device)
    #compiling i_frame_model
    if args.compile:
        print("Compiling the I-frame model...")
        i_frame_model = torch.compile(i_frame_model)

    print(
        f"Training model with lambda = {args.lambda_value}, quality_index = {args.quality_index}, stage = {args.stage}")

    # Initialize DCVC model
    model = DMC(lmbda=args.lambda_value)
    model = model.to(device)

    # Compile the model if specified
    if args.compile:
        print("Compiling the model...")
        model = torch.compile(model)

    # Initialize optimizer
    if args.stage == 4 and args.finetune:
        # Use lower learning rate for stage 4
        optimizer = optim.Adam(model.parameters(), lr=4e-5)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Initialize learning rate scheduler
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'multistep':
        milestones = [int(m) for m in args.lr_milestones.split(',')]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.learning_rate * args.lr_min_factor
        )
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_gamma, patience=args.lr_patience, verbose=True
        )
    else:  # 'none' or any other value
        scheduler = None

    # Create a warmup scheduler if warmup epochs are specified
    if args.lr_warmup_epochs > 0 and scheduler is not None and args.lr_scheduler != 'plateau':
        from torch.optim.lr_scheduler import LambdaLR
        
        # Create warmup scheduler
        def warmup_lambda(epoch):
            if epoch < args.lr_warmup_epochs:
                return epoch / args.lr_warmup_epochs
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
    else:
        warmup_scheduler = None

    # Initialize starting epoch and best loss
    start_epoch = 0
    best_loss = float('inf')
    best_loss_ema = float('inf')
    global_best_loss = float('inf')  # Track single global best regardless of weight type

    # Initialize EMA if specified
    ema = None
    if args.use_ema:
        print(f"Using EMA with decay rate: {args.ema_decay}")
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
        ema.to(device)

    # Log file
    stage_descriptions = {
        1: "Warm up MV generation part",
        2: "Train other modules",
        3: "Train with bit cost",
        4: "End-to-end training"
    }

    log_file = os.path.join(args.log_dir,
                            f'train_log_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_{args.model_type}.txt')

    # Check for SpyNet initialization
    if args.spynet_checkpoint:
        print(f"Initializing motion estimation network with pretrained SpyNet weights: {args.spynet_checkpoint}")
        spynet_checkpoint = torch.load(args.spynet_checkpoint, map_location=device)
        
        spynet_state_dict = spynet_checkpoint['state_dict']
        unwrap_model(model).optic_flow.load_state_dict(spynet_state_dict, strict=True)
        print("Loaded SpyNet weights directly into optic_flow component")
    
    if args.spynet_from_dcvc_checkpoint:
        print(f"Initializing motion estimation network with SpyNet weights from DCVC checkpoint: {args.spynet_from_dcvc_checkpoint}")
        spynet_checkpoint = torch.load(args.spynet_from_dcvc_checkpoint, map_location=device)

        # Extract only SpyNet weights
        spynet_state_dict = {}
        for key, value in spynet_checkpoint.items():
            if key.startswith('optic_flow.'):
                spynet_state_dict[key.replace('optic_flow.', '')] = value
        
        unwrap_model(model).optic_flow.load_state_dict(spynet_state_dict, strict=True)
        print("Loaded SpyNet weights from DCVC checkpoint into optic_flow component")

    # Resume training from checkpoint if specified
    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Check if this is a state_dict only or a complete checkpoint
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with training state
                unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
                best_loss = checkpoint['best_loss']

                if 'best_loss_ema' in checkpoint:
                    best_loss_ema = checkpoint['best_loss_ema']
                
                # Load global best loss if available
                if 'global_best_loss' in checkpoint:
                    global_best_loss = checkpoint['global_best_loss']
                
                # Load EMA state if it exists and EMA is enabled
                if args.use_ema and 'ema_state_dict' in checkpoint:
                    ema.load_state_dict(checkpoint['ema_state_dict'])
                    print("Loaded EMA state from checkpoint")
                
                # Load scheduler state if it exists and scheduler is initialized
                if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                
                print(f"Resumed from epoch {checkpoint['epoch']}, best loss: {best_loss:.6f}")
                if 'global_best_loss' in checkpoint:
                    print(f"Global best loss: {global_best_loss:.6f}")
                    with open(log_file, 'a') as f:
                        f.write(f"Global best loss: {global_best_loss:.6f}\n")
                        f.write(f"EMA best loss: {best_loss_ema:.6f}\n")
            else:
                # State dict only
                unwrap_model(model).load_state_dict(checkpoint)
                print("Loaded model weights only (no training state)")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting training from scratch")
    # Load from previous stage checkpoint if no resume but previous_stage_checkpoint is specified
    elif args.previous_stage_checkpoint:
        print(f"Loading model from previous stage checkpoint: {args.previous_stage_checkpoint}")
        try:
            checkpoint = torch.load(args.previous_stage_checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with training state
                unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model weights only (no training state)")
            else:
                unwrap_model(model).load_state_dict(checkpoint)  # Use load_state_dict method as defined in DCVC_net
            print("Successfully loaded model from previous stage")

            if args.not_skip_test: 
                # Evaluate based on training mode
                # For normal training, only use three-frame evaluation
                test_stats_three = evaluate_fully_batched_three(
                    model, i_frame_model, test_loader, device, args.stage-1,args
                )
                # Log results
                with open(log_file, 'a') as f:
                    f.write(f"From stage {args.stage-1}:\n")
                    f.write(f"  Test Loss Three: {test_stats_three['loss']:.6f}\n")
                    f.write(f"  Test MSE Three: {test_stats_three['mse']:.6f}\n")
                    f.write(f"  Test PSNR Three: {test_stats_three['psnr']:.4f}\n")
                    f.write(f"  Test BPP Three: {test_stats_three['bpp']:.6f}\n")
        except Exception as e:
            print(f"Error loading previous stage checkpoint: {e}")
            print("Starting training from scratch")
    
    if args.finetune_checkpoint and args.finetune:
        print(f"Finetuning from checkpoint: {args.finetune_checkpoint}")
        try:
            checkpoint = torch.load(args.finetune_checkpoint, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Full checkpoint with training state
                unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
                print("Loaded model weights only (no training state)")
            else:
                unwrap_model(model).load_state_dict(checkpoint)  # Use load_state_dict method as defined in DCVC_net
            print("Successfully loaded model from finetune checkpoint")
            
            if args.not_skip_test: 
                # Since we're in finetune mode, only use evaluate_fully_batched
                test_stats = evaluate_fully_batched(
                    model, i_frame_model, test_loader, device, args.stage, args.finetune, args
                )
                # Log results
                with open(log_file, 'a') as f:
                    f.write(f"From unfinetuned checkpoint:\n")
                    f.write(f"  Test Loss: {test_stats['loss']:.6f}\n")
                    f.write(f"  Test MSE: {test_stats['mse']:.6f}\n")
                    f.write(f"  Test PSNR: {test_stats['psnr']:.4f}\n")
                    f.write(f"  Test BPP: {test_stats['bpp']:.6f}\n")
        except Exception as e:
            print(f"Error loading finetune checkpoint: {e}")
            print("Starting training from scratch")

    with open(log_file, 'a') as f:
        f.write(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Lambda value: {args.lambda_value}\n")
        f.write(f"Quality index: {args.quality_index}\n")
        f.write(f"Model type: {args.model_type}\n")
        f.write(f"Stage: {args.stage} ({stage_descriptions[args.stage]})\n")
        f.write(f"I-frame model: {args.i_frame_model_path}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"LR Scheduler: {args.lr_scheduler}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Gradient accumulation steps: {args.gradient_accumulation_steps}\n")
        f.write(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}\n")
        if args.grad_clip_max_norm is not None:
            f.write(f"Gradient clipping max norm: {args.grad_clip_max_norm}\n")
        if args.use_ema:
            f.write(f"Using EMA with decay rate: {args.ema_decay}\n")
            if args.evaluate_both:
                f.write(f"Evaluating both normal and EMA weights\n")
        if args.global_best_dir:
            f.write(f"Tracking global best models in: {args.global_best_dir}\n")
        if args.lr_scheduler != 'none':
            if args.lr_scheduler == 'step':
                f.write(f"  Step size: {args.lr_step_size}, Gamma: {args.lr_gamma}\n")
            elif args.lr_scheduler == 'multistep':
                f.write(f"  Milestones: {args.lr_milestones}, Gamma: {args.lr_gamma}\n")
            elif args.lr_scheduler == 'cosine':
                f.write(f"  Min LR factor: {args.lr_min_factor}\n")
            elif args.lr_scheduler == 'plateau':
                f.write(f"  Patience: {args.lr_patience}, Factor: {args.lr_gamma}\n")
            if args.lr_warmup_epochs > 0:
                f.write(f"  Warmup epochs: {args.lr_warmup_epochs}\n")
        if args.previous_stage_checkpoint:
            f.write(f"Previous stage checkpoint: {args.previous_stage_checkpoint}\n")
        if args.resume:
            f.write(f"Resuming from checkpoint: {args.resume}\n")
            f.write(f"Starting from epoch: {start_epoch}\n")
        if args.spynet_checkpoint:
            f.write(f"SpyNet initialization: {args.spynet_checkpoint}\n")
        if args.spynet_from_dcvc_checkpoint:
            f.write(f"SpyNet initialization from DCVC checkpoint: {args.spynet_from_dcvc_checkpoint}\n")
        f.write(f"Training dataset: {len(train_dataset)} sequences\n")
        f.write(f"UVG dataset: {len(test_dataset)} sequences\n")
        f.write(f"Using fully batched GOP processing with parallel P-frame batch processing\n")
        f.write("=" * 80 + "\n")

    if args.reset_lr:
        print("Resetting learning rate to initial value")
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
            best_loss = float('inf')
            best_loss_ema = float('inf')
            global_best_loss = float('inf')  # Track single global best regardless of weight type
            #reset lr_scheduler state
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_gamma, patience=args.lr_patience, verbose=True
        )

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        # Record epoch start time
        epoch_start_time = time.time()
        epoch_start_str = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} - Learning rate: {current_lr:.6f}")
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{args.epochs} - Learning rate: {current_lr:.6f}\n")
            f.write(f"Epoch start time: {epoch_start_str}\n")

        # Apply warmup scheduler if in warmup phase
        if warmup_scheduler is not None and epoch < args.lr_warmup_epochs:
            warmup_scheduler.step()
        
        # Train one epoch with fully batched GOP processing
        train_stats = train_one_epoch_fully_batched(
            model, i_frame_model, train_loader, optimizer, device,
            args.stage, epoch + 1,  # Use 7 for Vimeo90k
            args.gradient_accumulation_steps, args.finetune, args.grad_clip_max_norm, ema,args
        )

        # Evaluate based on training mode (finetuning or normal training)
        if args.finetune:
            # Evaluate with current weights
            test_stats = evaluate_fully_batched(
                model, i_frame_model, test_loader, device, args.stage, args.finetune, args
            )
            
            # If using EMA and evaluate_both is enabled, also evaluate with EMA weights
            test_stats_normal = None
            if args.use_ema and args.evaluate_both:
                # Store current weights
                ema.store(model.parameters())
                
                # Copy EMA weights to model
                ema.copy_to(model.parameters())
                
                # Evaluate with EMA weights
                test_stats_ema = evaluate_fully_batched(
                    model, i_frame_model, test_loader, device, args.stage, args.finetune, args
                )
                
                # Restore original parameters
                ema.restore(model.parameters())
                
                # Rename test_stats to test_stats_normal for clarity
                test_stats_normal = test_stats
            
            test_stats_three = None  # Not using three-frame evaluation for finetuning
        else:
            # Using evaluate_fully_batched_three during normal training
            
            # Evaluate with current weights
            test_stats_three = evaluate_fully_batched_three(
                model, i_frame_model, test_loader, device, args.stage,args
            )
            
            # If using EMA and evaluate_both is enabled, also evaluate with EMA weights
            test_stats_three_normal = None
            if args.use_ema and args.evaluate_both:
                # Store current weights
                ema.store(model.parameters())
                
                # Copy EMA weights to model
                ema.copy_to(model.parameters())
                
                # Evaluate with EMA weights
                test_stats_three_ema = evaluate_fully_batched_three(
                    model, i_frame_model, test_loader, device, args.stage,args
                )
                
                # Restore original parameters
                ema.restore(model.parameters())
                
                # Rename test_stats_three to test_stats_three_normal for clarity
                test_stats_three_normal = test_stats_three
            
            test_stats = None  # Not using regular evaluation for normal training

        # Step scheduler after training (different for ReduceLROnPlateau)
        if scheduler is not None:
            if args.lr_scheduler == 'plateau':
                # Use the appropriate evaluation metric based on training mode
                if args.finetune:
                    scheduler.step(test_stats['loss'])
                else:
                    scheduler.step(test_stats_three['loss'])
            elif epoch >= args.lr_warmup_epochs:  # Only step main scheduler after warmup
                scheduler.step()

        # Record epoch end time and calculate duration
        epoch_end_time = time.time()
        epoch_end_str = time.strftime('%Y-%m-%d %H:%M:%S')
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Format duration as hours:minutes:seconds
        hours, remainder = divmod(epoch_duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

        # Log results
        with open(log_file, 'a') as f:
            f.write(f"Stage {args.stage}, Epoch {epoch + 1}/{args.epochs}:\n")
            f.write(f"  Train Loss: {train_stats['loss']:.6f}\n")
            f.write(f"  Train MSE: {train_stats['mse']:.6f}\n")
            f.write(f"  Train PSNR: {train_stats['psnr']:.4f}\n")
            f.write(f"  Train BPP: {train_stats['bpp']:.6f}\n")
            if 'bpp_y' in train_stats and train_stats['bpp_y'] > 0:
                f.write(f"  Train BPP_y: {train_stats['bpp_y']:.6f}\n")
            if 'bpp_z' in train_stats and train_stats['bpp_z'] > 0:
                f.write(f"  Train BPP_z: {train_stats['bpp_z']:.6f}\n")
            if 'bpp_mv_y' in train_stats and train_stats['bpp_mv_y'] > 0:
                f.write(f"  Train BPP_mv_y: {train_stats['bpp_mv_y']:.6f}\n")
            if 'bpp_mv_z' in train_stats and train_stats['bpp_mv_z'] > 0:
                f.write(f"  Train BPP_mv_z: {train_stats['bpp_mv_z']:.6f}\n")
            
            # Log appropriate evaluation metrics based on training mode
            if args.finetune:
                if args.use_ema and args.evaluate_both:
                    f.write(f"  Test Loss (Normal): {test_stats_normal['loss']:.6f}\n")
                    f.write(f"  Test MSE (Normal): {test_stats_normal['mse']:.6f}\n")
                    f.write(f"  Test PSNR (Normal): {test_stats_normal['psnr']:.4f}\n")
                    f.write(f"  Test BPP (Normal): {test_stats_normal['bpp']:.6f}\n")
                    
                    f.write(f"  Test Loss (EMA): {test_stats_ema['loss']:.6f}\n")
                    f.write(f"  Test MSE (EMA): {test_stats_ema['mse']:.6f}\n")
                    f.write(f"  Test PSNR (EMA): {test_stats_ema['psnr']:.4f}\n")
                    f.write(f"  Test BPP (EMA): {test_stats_ema['bpp']:.6f}\n")
                else:
                    f.write(f"  Test Loss: {test_stats['loss']:.6f}\n")
                    f.write(f"  Test MSE: {test_stats['mse']:.6f}\n")
                    f.write(f"  Test PSNR: {test_stats['psnr']:.4f}\n")
                    f.write(f"  Test BPP: {test_stats['bpp']:.6f}\n")
            else:
                if args.use_ema and args.evaluate_both:
                    f.write(f"  Test Loss Three (Normal): {test_stats_three_normal['loss']:.6f}\n")
                    f.write(f"  Test MSE Three (Normal): {test_stats_three_normal['mse']:.6f}\n")
                    f.write(f"  Test PSNR Three (Normal): {test_stats_three_normal['psnr']:.4f}\n")
                    f.write(f"  Test BPP Three (Normal): {test_stats_three_normal['bpp']:.6f}\n")
                    
                    f.write(f"  Test Loss Three (EMA): {test_stats_three_ema['loss']:.6f}\n")
                    f.write(f"  Test MSE Three (EMA): {test_stats_three_ema['mse']:.6f}\n")
                    f.write(f"  Test PSNR Three (EMA): {test_stats_three_ema['psnr']:.4f}\n")
                    f.write(f"  Test BPP Three (EMA): {test_stats_three_ema['bpp']:.6f}\n")
                else:
                    f.write(f"  Test Loss Three: {test_stats_three['loss']:.6f}\n")
                    f.write(f"  Test MSE Three: {test_stats_three['mse']:.6f}\n")
                    f.write(f"  Test PSNR Three: {test_stats_three['psnr']:.4f}\n")
                    f.write(f"  Test BPP Three: {test_stats_three['bpp']:.6f}\n")
                
            f.write(f"  Epoch end time: {epoch_end_str}\n")
            f.write(f"  Epoch duration: {duration_str} ({epoch_duration:.2f} seconds)\n")
            f.write("=" * 80 + "\n")

        # Get current test loss based on training mode and weight type
        if args.finetune:
            if args.use_ema and args.evaluate_both:
                current_test_loss_normal = test_stats_normal['loss']
                current_test_loss_ema = test_stats_ema['loss']
            else:
                current_test_loss = test_stats['loss']
                current_test_loss_ema = current_test_loss if args.use_ema else None
                current_test_loss_normal = current_test_loss
        else:
            if args.use_ema and args.evaluate_both:
                current_test_loss_normal = test_stats_three_normal['loss']
                current_test_loss_ema = test_stats_three_ema['loss']
            else:
                current_test_loss = test_stats_three['loss']
                current_test_loss_ema = current_test_loss if args.use_ema else None
                current_test_loss_normal = current_test_loss

        # Create save dictionary with all training state - DO THIS FIRST
        save_dict = {
            'epoch': epoch,
            'model_state_dict': unwrap_model(model).state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_stats['loss'],
            'best_loss': best_loss,
            'global_best_loss': global_best_loss,
            'stage': args.stage,
            'quality_index': args.quality_index,
            'lambda_value': args.lambda_value
        }
        
        # Add EMA state if enabled
        if args.use_ema:
            save_dict['ema_state_dict'] = ema.state_dict()
            save_dict['best_loss_ema'] = best_loss_ema
        
        # Add scheduler state if present
        if scheduler is not None:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()

        # Check if we have a new best model within this stage
        if current_test_loss_normal < best_loss:
            best_loss = current_test_loss_normal
            best_checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_best.pth'
            )
            # Update best loss in save_dict
            save_dict['best_loss'] = best_loss
            # Save full training state for the best model
            torch.save(save_dict, best_checkpoint_path)
            
            print(f"New best model saved with test loss: {best_loss:.6f}")
            with open(log_file, 'a') as f:
                f.write(f"New best model saved with test loss: {best_loss:.6f}\n")

        # Check for global best model (single best regardless of weight type)
        if args.global_best_dir:
            # Determine the current best loss (from either normal or EMA weights)
            current_best_loss = current_test_loss_normal
            is_ema_best = False
            
            # If using EMA, check if EMA weights are better
            if args.use_ema and current_test_loss_ema is not None and current_test_loss_ema < current_test_loss_normal:
                current_best_loss = current_test_loss_ema
                is_ema_best = True
            
            # Check if this is a new global best
            if current_best_loss < global_best_loss:
                global_best_loss = current_best_loss
                save_dict['global_best_loss'] = global_best_loss
                
                # Apply EMA weights if they're better
                if is_ema_best:
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                
                # Save global best checkpoint
                global_best_path = os.path.join(
                    args.global_best_dir,
                    f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_global_best.pth'
                )
                
                # Update model state dict in save_dict with current best weights
                save_dict['model_state_dict'] = unwrap_model(model).state_dict()
                save_dict['is_ema'] = is_ema_best  # Add flag to indicate if using EMA weights
                
                torch.save(save_dict, global_best_path)
                
                # Also save just the state dict for easy loading
                global_best_state_dict_path = os.path.join(
                    args.global_best_dir,
                    f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_global_best_state_dict.pth'
                )
                torch.save(unwrap_model(model).state_dict(), global_best_state_dict_path)
                
                # Restore original parameters if we applied EMA weights
                if is_ema_best:
                    ema.restore(model.parameters())
                
                weight_type = "EMA" if is_ema_best else "normal"
                print(f"New global best model saved with test loss: {global_best_loss:.6f} (using {weight_type} weights)")
                with open(log_file, 'a') as f:
                    f.write(f"New global best model saved with test loss: {global_best_loss:.6f} (using {weight_type} weights)\n")

        # If using EMA, also save a best model with EMA weights
        if args.use_ema:
            # Check if we have a new best EMA model
            if args.evaluate_both:
                if current_test_loss_ema < best_loss_ema:
                    best_loss_ema = current_test_loss_ema
                    
                    # Store current parameters and apply EMA weights
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    
                    # Save best EMA checkpoint
                    ema_best_checkpoint_path = os.path.join(
                        args.checkpoint_dir,
                        f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_best_ema.pth'
                    )
                    
                    # Create a new save_dict with EMA weights
                    ema_save_dict = save_dict.copy()
                    ema_save_dict['model_state_dict'] = unwrap_model(model).state_dict()
                    ema_save_dict['best_loss_ema'] = best_loss_ema
                    save_dict['best_loss_ema'] = best_loss_ema
                    
                    torch.save(ema_save_dict, ema_best_checkpoint_path)
                    
                    # Restore original parameters
                    ema.restore(model.parameters())
                    
                    print(f"New best EMA model saved with test loss: {best_loss_ema:.6f}")
                    with open(log_file, 'a') as f:
                        f.write(f"New best EMA model saved with test loss: {best_loss_ema:.6f}\n")
            else:
                # Just save the EMA version if it's a new best model overall
                if current_test_loss_normal < best_loss:
                    # Apply EMA weights
                    ema.store(model.parameters())
                    ema.copy_to(model.parameters())
                    
                    ema_best_checkpoint_path = os.path.join(
                        args.checkpoint_dir,
                        f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_best_ema.pth'
                    )
                    torch.save(unwrap_model(model).state_dict(), ema_best_checkpoint_path)
                    
                    # Restore original parameters
                    ema.restore(model.parameters())
                    
                    print(f"Best EMA model saved alongside best normal model")
                    with open(log_file, 'a') as f:
                        f.write(f"Best EMA model saved alongside best normal model\n")

        # Save latest checkpoint with training state for resuming - DO THIS LAST
        latest_checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_latest.pth'
        )
        
        # Make sure save_dict contains the updated global_best_loss
        save_dict['global_best_loss'] = global_best_loss
        
        # Save the checkpoint
        torch.save(save_dict, latest_checkpoint_path)
        
        print(f"Epoch {epoch + 1}/{args.epochs} completed. Latest checkpoint saved.")

    # Save final model for this stage (state_dict only for compatibility with original code)
    final_model_path = os.path.join(
        args.checkpoint_dir,
        f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}.pth'
    )
    
    # Save both regular and EMA versions of the final model if using EMA
    if args.use_ema:
        # Regular weights
        torch.save(unwrap_model(model).state_dict(), final_model_path)
        
        # EMA weights
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        ema_final_model_path = os.path.join(
            args.checkpoint_dir,
            f'model_dcvc_lambda_{args.lambda_value}_quality_{args.quality_index}_stage_{args.stage}_ema.pth'
        )
        torch.save(unwrap_model(model).state_dict(), ema_final_model_path)
        ema.restore(model.parameters())
        print(f"Final models for stage {args.stage} saved (both regular and EMA versions)")
    else:
        torch.save(unwrap_model(model).state_dict(), final_model_path)
        print(f"Final model for stage {args.stage} saved to {final_model_path}")

    # If this is the final stage (4), also save with the standard naming convention
    if args.stage == 4:
        if args.use_ema:
            # Apply EMA weights for the standard model
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            standard_model_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_quality_{args.quality_index}_{args.model_type}.pth'
            )
            torch.save(unwrap_model(model).state_dict(), standard_model_path)
            
            # Also save a version explicitly marked as EMA
            ema_standard_model_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_quality_{args.quality_index}_{args.model_type}_ema.pth'
            )
            torch.save(unwrap_model(model).state_dict(), ema_standard_model_path)
            ema.restore(model.parameters())
            print(f"Final model (standard name) saved with EMA weights")
        else:
            standard_model_path = os.path.join(
                args.checkpoint_dir,
                f'model_dcvc_quality_{args.quality_index}_{args.model_type}.pth'
            )
            torch.save(unwrap_model(model).state_dict(), standard_model_path)
            print(f"Final model (standard name) saved to {standard_model_path}")

    with open(log_file, 'a') as f:
        f.write(f"Training completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    print(f"Training completed for stage {args.stage}!")


if __name__ == '__main__':
    main()