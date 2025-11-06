# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time

import torch
from torch import nn

from .common_model import CompressionModel
from .video_net import ME_Spynet, flow_warp, ResBlock, bilineardownsacling, LowerBound, UNet, \
    get_enc_dec_models, get_hyper_enc_dec_models
from ..layers.layers import conv3x3, subpel_conv1x1, subpel_conv3x3
from ..utils.stream_helper import get_downsampled_shape, encode_p, decode_p, filesize, \
    get_rounded_q, get_state_dict


# First, add the LowerBound class and its supporting function
class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""
    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return torch.max(x, bound)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        pass_through_if = (x >= bound) | (grad_output < 0)
        return pass_through_if * grad_output, None


class LowerBound_sig(nn.Module):
    """Lower bound operator, computes `torch.max(x, bound)` with a custom
    gradient.
    
    The derivative is replaced by the identity function when `x` is moved
    towards the `bound`, otherwise the gradient is kept to zero.
    """
    bound: torch.Tensor
    
    def __init__(self, bound: float):
        super().__init__()
        self.register_buffer("bound", torch.Tensor([float(bound)]))
    
    @torch.jit.unused
    def lower_bound(self, x):
        return LowerBoundFunction.apply(x, self.bound)
    
    def forward(self, x):
        if torch.jit.is_scripting():
            return torch.max(x, self.bound)
        return self.lower_bound(x)

class FeatureExtractor(nn.Module):
    def __init__(self, channel=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, 3, stride=1, padding=1)
        self.res_block1 = ResBlock(channel)
        self.conv2 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block2 = ResBlock(channel)
        self.conv3 = nn.Conv2d(channel, channel, 3, stride=2, padding=1)
        self.res_block3 = ResBlock(channel)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class MultiScaleContextFusion(nn.Module):
    def __init__(self, channel_in=64, channel_out=64):
        super().__init__()
        self.conv3_up = subpel_conv3x3(channel_in, channel_out, 2)
        self.res_block3_up = ResBlock(channel_out)
        self.conv3_out = nn.Conv2d(channel_out, channel_out, 3, padding=1)
        self.res_block3_out = ResBlock(channel_out)
        self.conv2_up = subpel_conv3x3(channel_out * 2, channel_out, 2)
        self.res_block2_up = ResBlock(channel_out)
        self.conv2_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block2_out = ResBlock(channel_out)
        self.conv1_out = nn.Conv2d(channel_out * 2, channel_out, 3, padding=1)
        self.res_block1_out = ResBlock(channel_out)

    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        context2_up = self.conv2_up(torch.cat((context3_up, context2), dim=1))
        context2_up = self.res_block2_up(context2_up)
        context2_out = self.conv2_out(torch.cat((context3_up, context2), dim=1))
        context2_out = self.res_block2_out(context2_out)
        context1_out = self.conv1_out(torch.cat((context2_up, context1), dim=1))
        context1_out = self.res_block1_out(context1_out)
        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out
        return context1, context2, context3


class ContextualEncoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.conv1 = nn.Conv2d(channel_N + 3, channel_N, 3, stride=2, padding=1)
        self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.conv2 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.conv3 = nn.Conv2d(channel_N * 2, channel_N, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(channel_N, channel_M, 3, stride=2, padding=1)

    def forward(self, x, context1, context2, context3):
        feature = self.conv1(torch.cat([x, context1], dim=1))
        feature = self.res1(torch.cat([feature, context2], dim=1))
        feature = self.conv2(feature)
        feature = self.res2(torch.cat([feature, context3], dim=1))
        feature = self.conv3(feature)
        feature = self.conv4(feature)
        return feature


class ContextualDecoder(nn.Module):
    def __init__(self, channel_N=64, channel_M=96):
        super().__init__()
        self.up1 = subpel_conv3x3(channel_M, channel_N, 2)
        self.up2 = subpel_conv3x3(channel_N, channel_N, 2)
        self.res1 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up3 = subpel_conv3x3(channel_N * 2, channel_N, 2)
        self.res2 = ResBlock(channel_N * 2, bottleneck=True, slope=0.1,
                             start_from_relu=True, end_with_relu=True)
        self.up4 = subpel_conv3x3(channel_N * 2, 32, 2)

    def forward(self, x, context2, context3):
        feature = self.up1(x)
        feature = self.up2(feature)
        feature = self.res1(torch.cat([feature, context3], dim=1))
        feature = self.up3(feature)
        feature = self.res2(torch.cat([feature, context2], dim=1))
        feature = self.up4(feature)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self, ctx_channel=64, res_channel=32, channel=64):
        super().__init__()
        self.first_conv = nn.Conv2d(ctx_channel + res_channel, channel, 3, stride=1, padding=1)
        self.unet_1 = UNet(channel)
        self.unet_2 = UNet(channel)
        self.recon_conv = nn.Conv2d(channel, 3, 3, stride=1, padding=1)

    def forward(self, ctx, res):
        feature = self.first_conv(torch.cat((ctx, res), dim=1))
        feature = self.unet_1(feature)
        feature = self.unet_2(feature)
        recon = self.recon_conv(feature)
        return feature, recon


class DMC(CompressionModel):
    def __init__(self, anchor_num=4):
        super().__init__(y_distribution='laplace', z_channel=64, mv_z_channel=64)
        self.DMC_version = '1.19'

        channel_mv = 64
        channel_N = 64
        channel_M = 96

        self.channel_mv = channel_mv
        self.channel_N = channel_N
        self.channel_M = channel_M

        self.optic_flow = ME_Spynet()

        self.mv_encoder, self.mv_decoder = get_enc_dec_models(2, 2, channel_mv)
        self.mv_hyper_prior_encoder, self.mv_hyper_prior_decoder = \
            get_hyper_enc_dec_models(channel_mv, channel_N)

        self.mv_y_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, stride=1, padding=1)
        )

        self.mv_y_spatial_prior = nn.Sequential(
            nn.Conv2d(channel_mv * 4, channel_mv * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_mv * 3, channel_mv * 2, 3, padding=1)
        )

        self.feature_adaptor_I = nn.Conv2d(3, channel_N, 3, stride=1, padding=1)
        self.feature_adaptor_P = nn.Conv2d(channel_N, channel_N, 1)
        self.feature_extractor = FeatureExtractor()
        self.context_fusion_net = MultiScaleContextFusion()

        self.contextual_encoder = ContextualEncoder(channel_N=channel_N, channel_M=channel_M)

        self.contextual_hyper_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_M, channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel_N, channel_N, 3, stride=2, padding=1),
        )

        self.contextual_hyper_prior_decoder = nn.Sequential(
            conv3x3(channel_N, channel_M),
            nn.LeakyReLU(),
            subpel_conv1x1(channel_M, channel_M, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M, channel_M * 3 // 2),
            nn.LeakyReLU(),
            subpel_conv1x1(channel_M * 3 // 2, channel_M * 3 // 2, 2),
            nn.LeakyReLU(),
            conv3x3(channel_M * 3 // 2, channel_M * 2),
        )

        self.temporal_prior_encoder = nn.Sequential(
            nn.Conv2d(channel_N, channel_M * 3 // 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(channel_M * 3 // 2, channel_M * 2, 3, stride=2, padding=1),
        )

        self.y_prior_fusion = nn.Sequential(
            nn.Conv2d(channel_M * 5, channel_M * 4, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 4, channel_M * 3, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 3, 3, stride=1, padding=1)
        )

        self.y_spatial_prior = nn.Sequential(
            nn.Conv2d(channel_M * 4, channel_M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 3, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channel_M * 3, channel_M * 2, 3, padding=1)
        )

        self.contextual_decoder = ContextualDecoder(channel_N=channel_N, channel_M=channel_M)
        self.recon_generation_net = ReconGeneration()

        self.mv_y_q_basic = nn.Parameter(torch.ones((1, channel_mv, 1, 1)))
        self.mv_y_q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.y_q_basic = nn.Parameter(torch.ones((1, channel_M, 1, 1)))
        self.y_q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        self.anchor_num = int(anchor_num)
        self.mse = nn.MSELoss()
        # Add lower bound module with appropriate minimum scale value
        # 0.11 is a common minimum scale in image compression models
        self.lowerbound_scale = LowerBound_sig(0.11)

        self._initialize_weights()

    def multi_scale_feature_extractor(self, dpb):
        if dpb["ref_feature"] is None:
            feature = self.feature_adaptor_I(dpb["ref_frame"])
        else:
            feature = self.feature_adaptor_P(dpb["ref_feature"])
        return self.feature_extractor(feature)

    def motion_compensation(self, dpb, mv):
        warpframe = flow_warp(dpb["ref_frame"], mv)
        mv2 = bilineardownsacling(mv) / 2
        mv3 = bilineardownsacling(mv2) / 2
        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(dpb)
        context1 = flow_warp(ref_feature1, mv)
        context2 = flow_warp(ref_feature2, mv2)
        context3 = flow_warp(ref_feature3, mv3)
        context1, context2, context3 = self.context_fusion_net(context1, context2, context3)
        return context1, context2, context3, warpframe

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        y_q_scales = ckpt["y_q_scale"]
        mv_y_q_scales = ckpt["mv_y_q_scale"]
        return y_q_scales.reshape(-1), mv_y_q_scales.reshape(-1)

    def get_curr_mv_y_q(self, q_scale):
        q_basic = LowerBound.apply(self.mv_y_q_basic, 0.5)
        return q_basic * q_scale

    def get_curr_y_q(self, q_scale):
        q_basic = LowerBound.apply(self.y_q_basic, 0.5)
        return q_basic * q_scale

    def compress(self, x, dpb, mv_y_q_scale, y_q_scale):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        curr_mv_y_q = self.get_curr_mv_y_q(mv_y_q_scale)
        curr_y_q = self.get_curr_y_q(y_q_scale)

        est_mv = self.optic_flow(x, dpb["ref_frame"])
        mv_y = self.mv_encoder(est_mv)
        mv_y = mv_y / curr_mv_y_q
        mv_z = self.mv_hyper_prior_encoder(mv_y)
        mv_z_hat = torch.round(mv_z)
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            ref_mv_y = torch.zeros_like(mv_y)
        mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
        mv_q_step, mv_scales, mv_means = self.mv_y_prior_fusion(mv_params).chunk(3, 1)
        mv_y_q_w_0, mv_y_q_w_1, mv_scales_w_0, mv_scales_w_1, mv_y_hat = self.compress_dual_prior(
            mv_y, mv_means, mv_scales, mv_q_step, self.mv_y_spatial_prior)
        mv_y_hat = mv_y_hat * curr_mv_y_q

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, _ = self.motion_compensation(dpb, mv_hat)

        y = self.contextual_encoder(x, context1, context2, context3)
        y = y / curr_y_q
        z = self.contextual_hyper_prior_encoder(y)
        z_hat = torch.round(z)
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context3)
        ref_y = dpb["ref_y"]
        if ref_y is None:
            ref_y = torch.zeros_like(y)
        params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)

        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_q_w_0, y_q_w_1, scales_w_0, scales_w_1, y_hat = self.compress_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)
        y_hat = y_hat * curr_y_q

        recon_image_feature = self.contextual_decoder(y_hat, context2, context3)
        feature, x_hat = self.recon_generation_net(recon_image_feature, context1)

        self.entropy_coder.reset_encoder()
        _ = self.bit_estimator_z_mv.encode(mv_z_hat)
        _ = self.gaussian_encoder.encode(mv_y_q_w_0, mv_scales_w_0)
        _ = self.gaussian_encoder.encode(mv_y_q_w_1, mv_scales_w_1)
        _ = self.bit_estimator_z.encode(z_hat)
        _ = self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        _ = self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        bit_stream = self.entropy_coder.flush_encoder()

        result = {
            "dbp": {
                "ref_frame": x_hat,
                "ref_feature": feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
            },
            "bit_stream": bit_stream,
        }
        return result

    def decompress(self, dpb, string, height, width,
                   mv_y_q_scale, y_q_scale):
        curr_mv_y_q = self.get_curr_mv_y_q(mv_y_q_scale)
        curr_y_q = self.get_curr_y_q(y_q_scale)

        self.entropy_coder.set_stream(string)
        device = next(self.parameters()).device
        mv_z_size = get_downsampled_shape(height, width, 64)
        mv_z_hat = self.bit_estimator_z_mv.decode_stream(mv_z_size)
        mv_z_hat = mv_z_hat.to(device)
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            _, C, H, W = mv_params.size()
            ref_mv_y = torch.zeros((1, C // 2, H, W), device=mv_params.device)
        mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
        mv_q_step, mv_scales, mv_means = self.mv_y_prior_fusion(mv_params).chunk(3, 1)
        mv_y_hat = self.decompress_dual_prior(mv_means, mv_scales, mv_q_step,
                                              self.mv_y_spatial_prior)
        mv_y_hat = mv_y_hat * curr_mv_y_q

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, _ = self.motion_compensation(dpb, mv_hat)

        z_size = get_downsampled_shape(height, width, 64)
        z_hat = self.bit_estimator_z.decode_stream(z_size)
        z_hat = z_hat.to(device)
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context3)
        ref_y = dpb["ref_y"]
        if ref_y is None:
            _, C, H, W = temporal_params.size()
            ref_y = torch.zeros((1, C // 2, H, W), device=temporal_params.device)
        params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_hat = self.decompress_dual_prior(means, scales, q_step, self.y_spatial_prior)
        y_hat = y_hat * curr_y_q

        recon_image_feature = self.contextual_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)
        recon_image = recon_image.clamp(0, 1)

        return {
            "dpb": {
                "ref_frame": recon_image,
                "ref_feature": feature,
                "ref_y": y_hat,
                "ref_mv_y": mv_y_hat,
            },
        }

    def encode_decode(self, x, dpb, output_path=None,
                      pic_width=None, pic_height=None,
                      mv_y_q_scale=None, y_q_scale=None):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        if output_path is not None:
            mv_y_q_scale, mv_y_q_index = get_rounded_q(mv_y_q_scale)
            y_q_scale, y_q_index = get_rounded_q(y_q_scale)

            encoded = self.compress(x, dpb,
                                    mv_y_q_scale, y_q_scale)
            encode_p(encoded['bit_stream'], mv_y_q_index, y_q_index, output_path)
            bits = filesize(output_path) * 8
            mv_y_q_index, y_q_index, string = decode_p(output_path)

            start = time.time()
            decoded = self.decompress(dpb, string,
                                      pic_height, pic_width,
                                      mv_y_q_index / 100, y_q_index / 100)
            decoding_time = time.time() - start
            result = {
                "dpb": decoded["dpb"],
                "bit": bits,
                "decoding_time": decoding_time,
            }
            return result

        encoded = self.forward_one_frame(x, dpb,
                                         mv_y_q_scale=mv_y_q_scale, y_q_scale=y_q_scale,stage=4)
        result = {
            "dpb": encoded['dpb'],
            "bit_y": encoded['bit_y'].item(),
            "bit_z": encoded['bit_z'].item(),
            "bit_mv_y": encoded['bit_mv_y'].item(),
            "bit_mv_z": encoded['bit_mv_z'].item(),
            "bit": encoded['bit'].item(),
            "decoding_time": 0,
        }
        return result

    def lmb2qindex(self, lmbda):
        # lambda to q index. lmb[85, 170, 380, 840] --> [0, 1, 2, 3]
        #define a hash table to map lmbda to q index
        
        lmbda_to_q_index = {
            85: 0,
            170: 1,
            380: 2,
            840: 3
        }
        if isinstance(lmbda, int):
            if lmbda in lmbda_to_q_index:
                return lmbda_to_q_index[lmbda]
            else:
                raise ValueError("lmbda should be in [85, 170, 380, 840]")
        elif isinstance(lmbda, torch.Tensor):
            # for each element in lmbda, get the index
            lmbda_index = []
            for i in range(lmbda.shape[0]):
                if lmbda[i] in lmbda_to_q_index:
                    lmbda_index.append(lmbda_to_q_index[lmbda[i]])
                else:
                    raise ValueError("lmbda should be in [85, 170, 380, 840]")
            return lmbda_index
        else:
            raise ValueError("lmbda should be int or tensor")

    def qindex2qscale(self, qindex):
        if isinstance(qindex, int):
            # y_q_scale shape is (1, anchor_num, 1, 1), we use qindex to get the q_scale (1, 1, 1, 1)
            q_scale = self.y_q_scale[qindex:qindex+1,:,  :, :]
            mv_q_scale = self.mv_y_q_scale[qindex:qindex+1,:,  :, :]
            return mv_q_scale,q_scale
        elif isinstance(qindex, list):
            # Handle a list of indices for batch processing
            batch_size = len(qindex)
            
            # Convert list to tensor for indexing
            qindex_tensor = torch.tensor(qindex, device=self.y_q_scale.device)
            
            # Create batch indices
            batch_indices = torch.zeros(batch_size, dtype=torch.long, device=self.y_q_scale.device)
            
            # Select q_scale for each sample in the batch
            q_scale = self.y_q_scale[batch_indices, qindex_tensor, :, :]
            mv_q_scale = self.mv_y_q_scale[batch_indices, qindex_tensor, :, :]
            
            # Reshape to (batch_size, 1, 1, 1)
            q_scale = q_scale.view(batch_size, 1, 1, 1)
            mv_q_scale = mv_q_scale.view(batch_size, 1, 1, 1)
            
            return mv_q_scale,q_scale
        else:
            raise ValueError("qindex should be int or list")

    def forward_one_frame(self, x, dpb, mv_y_q_scale=None, y_q_scale=None,stage=1,lmbda=0.01):
        # lambda is either a scaler or a tensor shaped (B). 
        if isinstance(lmbda, int):
            self.lmbda = lmbda
        elif isinstance(lmbda, torch.Tensor):
            self.lmbda = lmbda
            #check shape B
            if lmbda.shape[0] != x.shape[0]:
                raise ValueError("lmbda shape should be B")
        else:
            raise ValueError("lmbda should be int or tensor")

        ref_frame = dpb["ref_frame"]

        if self.training:
            qindex = self.lmb2qindex(lmbda)
            mv_y_q_scale, y_q_scale = self.qindex2qscale(qindex)
            curr_mv_y_q = self.get_curr_mv_y_q(mv_y_q_scale)
            curr_y_q = self.get_curr_y_q(y_q_scale)

        else:
            curr_mv_y_q = self.get_curr_mv_y_q(mv_y_q_scale)
            curr_y_q = self.get_curr_y_q(y_q_scale)
        est_mv = self.optic_flow(x, ref_frame)
        mv_y = self.mv_encoder(est_mv)
        mv_y = mv_y / curr_mv_y_q
        mv_z = self.mv_hyper_prior_encoder(mv_y)
        mv_z_hat = self.quant(mv_z)
        mv_params = self.mv_hyper_prior_decoder(mv_z_hat)
        ref_mv_y = dpb["ref_mv_y"]
        if ref_mv_y is None:
            ref_mv_y = torch.zeros_like(mv_y)
        mv_params = torch.cat((mv_params, ref_mv_y), dim=1)
        mv_q_step, mv_scales, mv_means = self.mv_y_prior_fusion(mv_params).chunk(3, 1)
        mv_y_res, mv_y_q, mv_y_hat, mv_scales_hat = self.forward_dual_prior(
            mv_y, mv_means, mv_scales, mv_q_step, self.mv_y_spatial_prior)
        mv_y_hat = mv_y_hat * curr_mv_y_q

        mv_hat = self.mv_decoder(mv_y_hat)
        context1, context2, context3, warp_frame = self.motion_compensation(dpb, mv_hat)

        y = self.contextual_encoder(x, context1, context2, context3)
        y = y / curr_y_q
        z = self.contextual_hyper_prior_encoder(y)
        z_hat = self.quant(z)
        hierarchical_params = self.contextual_hyper_prior_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(context3)

        ref_y = dpb["ref_y"]
        if ref_y is None:
            ref_y = torch.zeros_like(y)
        params = torch.cat((temporal_params, hierarchical_params, ref_y), dim=1)
        q_step, scales, means = self.y_prior_fusion(params).chunk(3, 1)
        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)
        y_hat = y_hat * curr_y_q

        recon_image_feature = self.contextual_decoder(y_hat, context2, context3)
        feature, recon_image = self.recon_generation_net(recon_image_feature, context1)

        B, _, H, W = x.size()
        pixel_num =B * H * W

        if self.training:
            y_for_bit = self.add_noise(y_res)
            mv_y_for_bit = self.add_noise(mv_y_res)
            z_for_bit = self.add_noise(z)
            mv_z_for_bit = self.add_noise(mv_z)
        else:
            y_for_bit = y_q
            mv_y_for_bit = mv_y_q
            z_for_bit = z_hat
            mv_z_for_bit = mv_z_hat
        scales_hat = self.lowerbound_scale(scales_hat)
        mv_scales_hat = self.lowerbound_scale(mv_scales_hat)
        bits_y = self.get_y_laplace_bits_safe(y_for_bit, scales_hat)
        bits_mv_y = self.get_y_laplace_bits_safe(mv_y_for_bit, mv_scales_hat)
        bits_z = self.get_z_bits_safe(z_for_bit, self.bit_estimator_z)
        bits_mv_z = self.get_z_bits_safe(mv_z_for_bit, self.bit_estimator_z_mv)

        bpp_y = torch.sum(bits_y) / pixel_num
        bpp_z = torch.sum(bits_z) / pixel_num
        bpp_mv_y = torch.sum(bits_mv_y) / pixel_num
        bpp_mv_z = torch.sum(bits_mv_z) / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        bit = bpp * pixel_num

        loss = 0
        if stage == 1:
            #in stage 1, we calculate L_me = lambda*distortion(rec,inp) + bpp_mv_y + bpp_mv_z
            mse_loss = self.mse(warp_frame, x)
            distortion = self.lmbda * mse_loss
            L_me = distortion + bpp_mv_y + bpp_mv_z
            bpp_train = bpp_mv_y + bpp_mv_z
            loss = L_me
        elif stage == 2:
            #in stage 2, we train other modules except mv generation module. at this time, we freeze the mv generation module and calculate L_rec = lambda*distortion(rec,inp)
            mse_loss = self.mse(recon_image, x)
            L_rec = self.lmbda * mse_loss
            bpp_train = torch.tensor(0).to(x.device)
            loss = L_rec
        elif stage == 3:
            #in stage 3, the mv generation module is still frozen, and we calculate L_con = lambda*distortion(rec,inp) + bpp_y + bpp_z
            mse_loss = self.mse(recon_image, x)
            distortion = self.lmbda * mse_loss
            L_con = distortion + bpp_y + bpp_z
            bpp_train = bpp_y + bpp_z
            loss = L_con
        elif stage == 4:
            #in stage 4, we train all modules and calculate L_all = lambda*distortion(rec,inp) + bpp
            mse_loss = self.mse(recon_image, x)
            distortion = self.lmbda * mse_loss
            L_all = distortion + bpp
            bpp_train = bpp
            loss = L_all
        bit = torch.sum(bpp) * pixel_num
        bit_y = torch.sum(bpp_y) * pixel_num
        bit_z = torch.sum(bpp_z) * pixel_num
        bit_mv_y = torch.sum(bpp_mv_y) * pixel_num
        bit_mv_z = torch.sum(bpp_mv_z) * pixel_num
        

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "pixel_rec": warp_frame,
                "ref_mv_y": mv_y_hat,
                "dpb": {
                    "ref_frame": recon_image,
                    "ref_feature": feature,
                    "ref_y": y_hat,
                    "ref_mv_y": mv_y_hat,
                },
                "bit": bit,
                "loss": loss,
                "mse_loss": mse_loss,
                "bpp_train": bpp_train,
                "bit": bit,
                "bit_y": bit_y,
                "bit_z": bit_z,
                "bit_mv_y": bit_mv_y,
                "bit_mv_z": bit_mv_z,
                }

    def forward(self, x, dpb, mv_y_q_scale=None, y_q_scale=None,stage=1,lmbda= None):
        return self.forward_one_frame(x, dpb,
                                      mv_y_q_scale=mv_y_q_scale, y_q_scale=y_q_scale,stage=stage,lmbda=lmbda)
