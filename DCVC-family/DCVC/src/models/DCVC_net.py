import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .video_net import ME_Spynet, GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1
from ..entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from ..utils.stream_helper import get_downsampled_shape
from ..layers.layers import MaskedConv2d, subpel_conv3x3


class DCVC_net(nn.Module):
    def __init__(self, lmbda=1.0):
        super().__init__()
        out_channel_mv = 128
        out_channel_N = 64
        out_channel_M = 96

        self.lmbda = lmbda

        self.out_channel_mv = out_channel_mv
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_z_mv = BitEstimator(out_channel_N)

        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        self.context_refine = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.gaussian_encoder = GaussianEncoder()

        self.mvEncoder = nn.Sequential(
            nn.Conv2d(2, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
        )

        self.mvDecoder_part1 = nn.Sequential(
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, 2, 3, stride=2, padding=1, output_padding=1),
        )

        self.mvDecoder_part2 = nn.Sequential(
            nn.Conv2d(5, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
        )

        self.contextualEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N+3, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )

        self.contextualDecoder_part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
        )

        self.contextualDecoder_part2 = nn.Sequential(
            nn.Conv2d(out_channel_N*2, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        self.priorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )

        self.priorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_N, out_channel_M, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 3, stride=1, padding=1)
        )

        self.mvpriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_mv, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )

        self.mvpriorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_N, out_channel_N, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N, out_channel_N * 3 // 2, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N * 3 // 2, out_channel_mv*2, 3, stride=1, padding=1)
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(out_channel_M * 12 // 3, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M * 6 // 3, 1),
        )

        self.auto_regressive = MaskedConv2d(
            out_channel_M, 2 * out_channel_M, kernel_size=5, padding=2, stride=1
        )

        self.auto_regressive_mv = MaskedConv2d(
            out_channel_mv, 2 * out_channel_mv, kernel_size=5, padding=2, stride=1
        )

        self.entropy_parameters_mv = nn.Sequential(
            nn.Conv2d(out_channel_mv * 12 // 3, out_channel_mv * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 10 // 3, out_channel_mv * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 8 // 3, out_channel_mv * 6 // 3, 1),
        )

        self.temporalPriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )
        self.mse = nn.MSELoss()

        self.opticFlow = ME_Spynet()

    def motioncompensation(self, ref, mv):
        ref_feature = self.feature_extract(ref)
        prediction_init = flow_warp(ref_feature, mv)
        context = self.context_refine(prediction_init)

        return context
    
    def pixel_motioncompensation(self, ref, mv):
        warped = flow_warp(ref, mv)
        return warped

    def mv_refine(self, ref, mv):
        return self.mvDecoder_part2(torch.cat((mv, ref), 1)) + mv

    def quantize(self, inputs, mode, means=None):
        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        outputs = inputs.clone()
        if means is not None:
            outputs -= means
        outputs = torch.round(outputs)
        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs
        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    def feature_probs_based_sigma(self, feature, mean, sigma, training=True, bin_size=1.0, prob_clamp=1e-6,noise = None):
        """
        A numerically stable version of the feature probability calculation based on sigma.

        Args:
            feature (Tensor): Input feature tensor
            mean (Tensor): Mean tensor for quantization
            sigma (Tensor): Scale parameter for Laplace distribution
            training (bool): Whether in training mode. If True, uses "noise" quantization.
            bin_size (float): Quantization bin size, typically 1.0
            prob_clamp (float): Threshold for numerical stability approximation
        """

        # Ensure proper calculation precision
        @torch.no_grad()  # We don't need gradient through the assert check
        def check_sigma(s):
            assert s.min() > 0, f"Invalid sigma value: {s.min()}"
            return s

        # Convert all inputs to float32 to ensure precision
        feature = feature.float()
        mean = mean.float() if mean is not None else mean
        sigma = check_sigma(sigma.float())

        # Apply quantization based on training mode
        if training:
            if noise is not None:
                outputs = feature + noise
            else:
                raise ValueError("Noise tensor is required for training mode")
        else:
            outputs = self.quantize(
                feature,  "dequantize", mean
            )

        # Compute centered values
        values = outputs - mean

        # Clamp sigma to ensure numerical stability
        sigma = sigma.clamp(1e-5, 1e10)

        # Create Laplace distribution with zero mean and sigma scale
        mu = torch.zeros_like(sigma)
        laplace_dist = torch.distributions.laplace.Laplace(mu, sigma)

        # Safe log probability mass calculation
        def safe_log_prob_mass(dist, x, bin_size, prob_clamp):
            # Calculate probability mass: CDF(x+0.5) - CDF(x-0.5)
            prob_mass = dist.cdf(x + 0.5 * bin_size) - dist.cdf(x - 0.5 * bin_size)

            # Use approximation for numerical stability when probability mass is small
            log_prob = torch.where(
                prob_mass > prob_clamp,
                torch.log(torch.clamp(prob_mass, min=1e-10)),
                # Use log of PDF times bin size as approximation
                # For Laplace distribution, this is a good approximation
                dist.log_prob(x) + math.log(bin_size)
            )
            return log_prob, prob_mass

        # Calculate log probability and probability mass
        log_probs, probs = safe_log_prob_mass(laplace_dist, values, bin_size, prob_clamp)

        # Convert from nats to bits and sum
        total_bits = torch.sum(torch.clamp(-log_probs / math.log(2.0), 0, 50))

        return total_bits, probs

    def iclr18_estrate_bits_z(self, z, prob_clamp=1e-6):
        """
        A numerically stable bit rate estimation function based on ICLR'18 method.

        Args:
            z (Tensor): Input tensor
            prob_clamp (float): Threshold for numerical stability

        Returns:
            total_bits (Tensor): Estimated total bits
            prob (Tensor): Probability mass function values
        """
        # Calculate probability mass function (P(z-0.5 < Z <= z+0.5))
        prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)

        # Ensure computation precision
        prob = prob.float()

        # Calculate safe log probability
        def safe_log_prob(p, eps=1e-10):
            # Direct log calculation when probability is above threshold
            # Otherwise use Laplacian approximation for numerical stability
            log_p = torch.where(
                p > prob_clamp,
                torch.log(torch.clamp(p, min=eps)),
                # Using properties of Laplace distribution for approximation
                # Laplace distribution has exponential decay in tails, so linear approximation works well
                # This is consistent with the BitEstimator model characteristics
                torch.log(torch.tensor(prob_clamp, device=p.device, dtype=p.dtype)) + (p - prob_clamp) / prob_clamp
            )
            return log_p

        # Calculate information content (bits)
        log_prob = safe_log_prob(prob)
        bits = -log_prob / math.log(2.0)  # Convert to base-2 logarithm (bits)

        # Limit extreme values to prevent gradient explosion
        total_bits = torch.sum(torch.clamp(bits, 0, 50))

        return total_bits, prob

    def iclr18_estrate_bits_z_mv(self, z_mv, prob_clamp=1e-6):
        """
        A numerically stable bit rate estimation function for motion vectors based on ICLR'18 method.

        Args:
            z_mv (Tensor): Input motion vector tensor
            prob_clamp (float): Threshold for numerical stability

        Returns:
            total_bits (Tensor): Estimated total bits
            prob (Tensor): Probability mass function values
        """
        # Calculate probability mass function (P(z_mv-0.5 < Z <= z_mv+0.5))
        prob = self.bitEstimator_z_mv(z_mv + 0.5) - self.bitEstimator_z_mv(z_mv - 0.5)

        # Ensure computation precision
        prob = prob.float()

        # Calculate safe log probability
        def safe_log_prob(p, eps=1e-10):
            # Direct log calculation when probability is above threshold
            # Otherwise use approximation for numerical stability
            log_p = torch.where(
                p > prob_clamp,
                torch.log(torch.clamp(p, min=eps)),
                # Using first-order Taylor expansion approximation of log around prob_clamp
                # This is appropriate for the BitEstimator model
                torch.log(torch.tensor(prob_clamp, device=p.device, dtype=p.dtype)) + (p - prob_clamp) / prob_clamp
            )
            return log_p

        # Calculate information content (bits)
        log_prob = safe_log_prob(prob)
        bits = -log_prob / math.log(2.0)  # Convert to base-2 logarithm (bits)

        # Limit extreme values to prevent gradient explosion
        total_bits = torch.sum(torch.clamp(bits, 0, 50))

        return total_bits, prob

    def update(self, force=False):
        self.bitEstimator_z_mv.update(force=force)
        self.bitEstimator_z.update(force=force)
        self.gaussian_encoder.update(force=force)

    def encode_decode(self, ref_frame, input_image, output_path):
        encoded = self.encode(ref_frame, input_image, output_path)
        decoded = self.decode(ref_frame, output_path)
        encoded['recon_image'] = decoded
        return encoded

    def encode(self, ref_frame, input_image, output_path):
        from ..utils.stream_helper import encode_p
        N, C, H, W = ref_frame.size()
        compressed = self.compress(ref_frame, input_image)
        mv_y_string = compressed['mv_y_string']
        mv_z_string = compressed['mv_z_string']
        y_string = compressed['y_string']
        z_string = compressed['z_string']
        encode_p(H, W, mv_y_string, mv_z_string, y_string, z_string, output_path)
        return {
            'bpp_mv_y': compressed['bpp_mv_y'],
            'bpp_mv_z': compressed['bpp_mv_z'],
            'bpp_y': compressed['bpp_y'],
            'bpp_z': compressed['bpp_z'],
            'bpp': compressed['bpp'],
        }

    def decode(self, ref_frame, input_path):
        from ..utils.stream_helper import decode_p
        height, width, mv_y_string, mv_z_string, y_string, z_string = decode_p(input_path)
        return self.decompress(ref_frame, mv_y_string, mv_z_string,
                               y_string, z_string, height, width)

    def compress_ar(self, y, kernel_size, context_prediction, params, entropy_parameters):
        kernel_size = 5
        padding = (kernel_size - 1) // 2

        height = y.size(2)
        width = y.size(3)

        y_hat = F.pad(y, (padding, padding, padding, padding))
        y_q = torch.zeros_like(y)
        y_scales = torch.zeros_like(y)

        for h in range(height):
            for w in range(width):
                y_crop = y_hat[0:1, :, h:h + kernel_size, w:w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    context_prediction.weight,
                    bias=context_prediction.bias,
                )

                p = params[0:1, :, h:h + 1, w:w + 1]
                gaussian_params = entropy_parameters(torch.cat((p, ctx_p), dim=1))
                means_hat, scales_hat = gaussian_params.chunk(2, 1)

                y_crop = y_crop[0:1, :, padding:padding+1, padding:padding+1]
                y_crop_q = torch.round(y_crop - means_hat)
                y_hat[0, :, h + padding, w + padding] = (y_crop_q + means_hat)[0, :, 0, 0]
                y_q[0, :, h, w] = y_crop_q[0, :, 0, 0]
                y_scales[0, :, h, w] = scales_hat[0, :, 0, 0]
        # change to channel last
        y_q = y_q.permute(0, 2, 3, 1)
        y_scales = y_scales.permute(0, 2, 3, 1)
        y_string = self.gaussian_encoder.compress(y_q, y_scales)
        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        return y_string, y_hat

    def decompress_ar(self, y_string, channel, height, width, downsample, kernel_size,
                      context_prediction, params, entropy_parameters):
        device = next(self.parameters()).device
        padding = (kernel_size - 1) // 2

        y_size = get_downsampled_shape(height, width, downsample)
        y_height = y_size[0]
        y_width = y_size[1]

        y_hat = torch.zeros(
            (1, channel, y_height + 2 * padding, y_width + 2 * padding),
            device=params.device,
        )

        self.gaussian_encoder.set_stream(y_string)

        for h in range(y_height):
            for w in range(y_width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[0:1, :, h:h + kernel_size, w:w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    context_prediction.weight,
                    bias=context_prediction.bias,
                )
                p = params[0:1, :, h:h + 1, w:w + 1]
                gaussian_params = entropy_parameters(torch.cat((p, ctx_p), dim=1))
                means_hat, scales_hat = gaussian_params.chunk(2, 1)
                rv = self.gaussian_encoder.decode_stream(scales_hat)
                rv = rv.to(device)
                rv = rv + means_hat
                y_hat[0, :, h + padding: h + padding + 1, w + padding: w + padding + 1] = rv

        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        return y_hat

    def compress(self, referframe, input_image):
        device = input_image.device
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        z_mv = self.mvpriorEncoder(mvfeature)
        compressed_z_mv = torch.round(z_mv)
        mv_z_string = self.bitEstimator_z_mv.compress(compressed_z_mv)
        mv_z_size = [compressed_z_mv.size(2), compressed_z_mv.size(3)]
        mv_z_hat = self.bitEstimator_z_mv.decompress(mv_z_string, mv_z_size)
        mv_z_hat = mv_z_hat.to(device)

        params_mv = self.mvpriorDecoder(mv_z_hat)
        mv_y_string, mv_y_hat = self.compress_ar(mvfeature, 5, self.auto_regressive_mv,
                                                 params_mv, self.entropy_parameters_mv)

        quant_mv_upsample = self.mvDecoder_part1(mv_y_hat)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        context = self.motioncompensation(referframe, quant_mv_upsample_refine)

        temporal_prior_params = self.temporalPriorEncoder(context)
        feature = self.contextualEncoder(torch.cat((input_image, context), dim=1))
        z = self.priorEncoder(feature)
        compressed_z = torch.round(z)
        z_string = self.bitEstimator_z.compress(compressed_z)
        z_size = [compressed_z.size(2), compressed_z.size(3)]
        z_hat = self.bitEstimator_z.decompress(z_string, z_size)
        z_hat = z_hat.to(device)

        params = self.priorDecoder(z_hat)
        y_string, y_hat = self.compress_ar(feature, 5, self.auto_regressive,
                                           torch.cat((temporal_prior_params, params), dim=1), self.entropy_parameters)

        recon_image_feature = self.contextualDecoder_part1(y_hat)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context), dim=1))

        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = len(y_string) * 8 / pixel_num
        bpp_z = len(z_string) * 8 / pixel_num
        bpp_mv_y = len(mv_y_string) * 8 / pixel_num
        bpp_mv_z = len(mv_z_string) * 8 / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "recon_image": recon_image,
                "mv_y_string": mv_y_string,
                "mv_z_string": mv_z_string,
                "y_string": y_string,
                "z_string": z_string,
                }

    def decompress(self, referframe, mv_y_string, mv_z_string, y_string, z_string, height, width):
        device = next(self.parameters()).device
        mv_z_size = get_downsampled_shape(height, width, 64)
        mv_z_hat = self.bitEstimator_z_mv.decompress(mv_z_string, mv_z_size)
        mv_z_hat = mv_z_hat.to(device)
        params_mv = self.mvpriorDecoder(mv_z_hat)
        mv_y_hat = self.decompress_ar(mv_y_string, self.out_channel_mv, height, width, 16, 5,
                                      self.auto_regressive_mv, params_mv,
                                      self.entropy_parameters_mv)

        quant_mv_upsample = self.mvDecoder_part1(mv_y_hat)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        context = self.motioncompensation(referframe, quant_mv_upsample_refine)
        temporal_prior_params = self.temporalPriorEncoder(context)

        z_size = get_downsampled_shape(height, width, 64)
        z_hat = self.bitEstimator_z.decompress(z_string, z_size)
        z_hat = z_hat.to(device)
        params = self.priorDecoder(z_hat)
        y_hat = self.decompress_ar(y_string, self.out_channel_M, height, width, 16, 5,
                                   self.auto_regressive, torch.cat((temporal_prior_params, params), dim=1),
                                   self.entropy_parameters)
        recon_image_feature = self.contextualDecoder_part1(y_hat)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context), dim=1))
        recon_image = recon_image.clamp(0, 1)

        return recon_image

    def forward(self, referframe, input_image,training = True,stage = 1):
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        z_mv = self.mvpriorEncoder(mvfeature)
        # compressed_z_mv = torch.round(z_mv)
        if training == True:
            compressed_z_mv = z_mv + torch.empty_like(z_mv).uniform_(-0.5, 0.5)
        else:
            compressed_z_mv = torch.round(z_mv)

        params_mv = self.mvpriorDecoder(compressed_z_mv)

        # quant_mv = torch.round(mvfeature)
        if training == True:
            quant_mv_noise = torch.empty_like(mvfeature).uniform_(-0.5, 0.5)
            quant_mv = mvfeature + quant_mv_noise
        else:
            quant_mv = torch.round(mvfeature)

        ctx_params_mv = self.auto_regressive_mv(quant_mv)
        gaussian_params_mv = self.entropy_parameters_mv(
            torch.cat((params_mv, ctx_params_mv), dim=1)
        )
        means_hat_mv, scales_hat_mv = gaussian_params_mv.chunk(2, 1)
        scales_hat_mv = torch.nn.functional.softplus(scales_hat_mv+2.3)-2.3 #make logscale > -2.3
        scales_hat_mv = torch.exp(scales_hat_mv)

        quant_mv_upsample = self.mvDecoder_part1(quant_mv)

        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)

        context = self.motioncompensation(referframe, quant_mv_upsample_refine)
        pixel_rec = self.pixel_motioncompensation(referframe, quant_mv_upsample_refine)

        temporal_prior_params = self.temporalPriorEncoder(context)

        feature = self.contextualEncoder(torch.cat((input_image, context), dim=1))
        z = self.priorEncoder(feature)
        # compressed_z = torch.round(z)
        if training == True:
            compressed_z = z + torch.empty_like(z).uniform_(-0.5, 0.5)
        else:
            compressed_z = torch.round(z)
        params = self.priorDecoder(compressed_z)

        feature_renorm = feature

        # compressed_y_renorm = torch.round(feature_renorm)
        if training == True:
            compressed_y_renorm_noise = torch.empty_like(feature_renorm).uniform_(-0.5, 0.5)
            compressed_y_renorm = feature_renorm + compressed_y_renorm_noise
        else:
            compressed_y_renorm = torch.round(feature_renorm)

        ctx_params = self.auto_regressive(compressed_y_renorm)
        gaussian_params = self.entropy_parameters(
            torch.cat((temporal_prior_params, params, ctx_params), dim=1)
        )
        means_hat, scales_hat = gaussian_params.chunk(2, 1)
        scales_hat = torch.nn.functional.softplus(scales_hat+2.3)-2.3 #make logscale > -2.3
        scales_hat = torch.exp(scales_hat)

        recon_image_feature = self.contextualDecoder_part1(compressed_y_renorm)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context), dim=1))

        total_bits_y, _ = self.feature_probs_based_sigma(
            feature_renorm, means_hat, scales_hat,training=training,noise = compressed_y_renorm_noise if training else None)
        total_bits_mv, _ = self.feature_probs_based_sigma(mvfeature, means_hat_mv, scales_hat_mv,training=training, noise = quant_mv_noise if training else None)
        total_bits_z, _ = self.iclr18_estrate_bits_z(compressed_z)
        total_bits_z_mv, _ = self.iclr18_estrate_bits_z_mv(compressed_z_mv)

        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bpp_mv_y = total_bits_mv / pixel_num
        bpp_mv_z = total_bits_z_mv / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

        loss = 0
        #loss calculation
        if stage == 1:
            #in stage 1, we calculate L_me = lambda*distortion(rec,inp) + bpp_mv_y + bpp_mv_z
            mse_loss = self.mse(pixel_rec, input_image)
            distortion = self.lmbda *mse_loss
            L_me = distortion + bpp_mv_y + bpp_mv_z
            bpp_train = bpp_mv_y + bpp_mv_z
            loss = L_me
        elif stage == 2:
            #in stage 2, we train other modules except mv generation module. at this time, we freeze the mv generation module and calculate L_rec = lambda*distortion(rec,inp)
            mse_loss = self.mse(recon_image, input_image)
            L_rec = self.lmbda *mse_loss
            bpp_train = torch.tensor(0)
            loss = L_rec
        elif stage ==3:
            #in stage 3, the mv generation module is still frozen, and we calculate L_con = lambda*distortion(rec,inp) + bpp_y + bpp_z
            mse_loss = self.mse(recon_image, input_image)
            distortion = self.lmbda *mse_loss
            L_con = distortion + bpp_y + bpp_z
            bpp_train = bpp_y + bpp_z
            loss = L_con
        elif stage == 4:
            #in stage 4, we train all modules and calculate L_all = lambda*distortion(rec,inp) + bpp
            mse_loss = self.mse(recon_image, input_image)
            distortion = self.lmbda *mse_loss
            L_all = distortion + bpp
            bpp_train = bpp
            loss = L_all


        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "recon_image": recon_image,
                "context": context,
                "loss": loss,
                "mse_loss": mse_loss,
                "bpp_train": bpp_train,
                "pixel_rec": pixel_rec,
                }

    def load_dict(self, pretrained_dict):
        result_dict = {}
        for key, weight in pretrained_dict.items():
            result_key = key
            if key[:7] == "module.":
                result_key = key[7:]
            result_dict[result_key] = weight

        self.load_state_dict(result_dict)
