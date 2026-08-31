import numpy as np
import random
import torch
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.data.transforms import paired_random_crop
from .sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from torch.nn import functional as F
from collections import OrderedDict
from basicsr.utils.dist_util import master_only
import os
import os.path as osp
from basicsr.utils import get_root_logger, tensor2img, imwrite
from basicsr.metrics import calculate_metric
import pyiqa
import time
from tqdm import tqdm
import corun_colabator.archs.open_clip as open_clip
import corun_colabator.archs.memory_bank as memory_bank
from corun_colabator.archs.corun_arch import enable_mc_dropout, disable_mc_dropout
from corun_colabator.archs.quadtree_router import QuadtreeRouter, joint_uncertainty

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).to(self.device)

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


@MODEL_REGISTRY.register()
class Colabator_by_Depth(SRModel):
    """
    It is trained without GAN losses.
    It mainly performs:
    1. randomly synthesize LQ images in GPU tensors
    2. optimize the networks with GAN training.
    """

    def __init__(self, opt):
        super(Colabator_by_Depth, self).__init__(opt)
        if self.is_train:
            self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
            if self.mixing_flag:
                mixup_beta = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
                use_identity = self.opt['train']['mixing_augs'].get('use_identity', False)
                self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        if self.is_train:
            self.block_size = self.opt['colabator'].get('block_size', None)
            if self.opt['colabator'].get('use_clip', False) or self.opt['train'].get('use_clip_loss', False):
                self.init_clip()
            if self.opt['colabator'].get('use_nr_iqa', False):
                self.init_nriqa()
            self.init_mmb()
            self.init_lucid()

    def init_lucid(self):
        """Set up the LUCID additions: MC-Dropout sampling and the quadtree router.

        Every knob is read from the `colabator` option block so the values in the
        paper are visible in the config rather than buried in the code.
        """
        cfg = self.opt['colabator']
        # Teacher-only MC Dropout (paper Sec. "Teacher-only MC Dropout").
        self.mc_K = int(cfg.get('mc_K', 5))
        # ASM gate threshold tau_ASM (paper Eq. 5).
        self.tau_asm = float(cfg.get('uncertainty_threshold', 0.05))
        # Joint quadtree spatial routing (paper Sec. "Joint Quadtree Spatial Routing").
        self.use_quadtree = bool(cfg.get('use_quadtree', True))
        self.quadtree_router = None
        if self.use_quadtree:
            self.quadtree_router = QuadtreeRouter(
                scorer=self.score_blocks,
                sizes=cfg.get('quadtree_sizes', [64, 32, 16, 8]),
                tau_q=float(cfg.get('tau_q', 0.3)),
                tau_crit=float(cfg.get('tau_crit', 0.7)),
            )

    def score_blocks(self, blocks):
        """Combined DA-CLIP + MUSIQ reliability weight for a batch of crops.

        Mirrors the arithmetic that `labal_selection` applies to its uniform
        grid, so the router's per-leaf weight is directly comparable to the flat
        mask it replaces:

            M_leaf = (nr_iqa_term + clip_term) / (len(degradation_type) + 1)

        Args:
            blocks (Tensor): [N, 3, h, w] crops in [0, 1].

        Returns:
            Tensor: [N] weights, later clamped to [0, 1] by the router.
        """
        with torch.no_grad():
            if self.opt['colabator'].get('use_nr_iqa', False):
                nr = self.nr_iqa(blocks).reshape(-1)
                nr = (nr - self.nr_iqa_scale[0]) / (self.nr_iqa_scale[1] - self.nr_iqa_scale[0])
                if self.nr_iqa_better != 'higher':
                    nr = 1 - nr
            else:
                nr = torch.zeros(blocks.shape[0], device=blocks.device)

            if self.opt['colabator'].get('use_clip', False):
                n_degr = len(self.degradation_type)
                clip_term = n_degr - self.get_clip_degrad_rate(blocks).reshape(-1)
                if self.clip_better != 'higher':
                    clip_term = n_degr - clip_term
            else:
                n_degr = len(self.degradation_type) if self.degradation_type else 0
                clip_term = torch.zeros(blocks.shape[0], device=blocks.device)

            return (nr + clip_term) / (n_degr + 1)

    @torch.no_grad()
    def mc_teacher_forward(self, real):
        """Draw K stochastic teacher samples (paper Eq. 6-7).

        Dropout is enabled only on the teacher's dropout layers; every other
        layer stays in eval(). The student is never sampled this way, so the
        deployed generator is deterministic and inference cost is unchanged.

        Returns:
            tuple: (J_mu, T_mu, var_J, var_T). With mc_K <= 1 the variances are
            zero and the means reduce to a single deterministic pass.
        """
        net = self.net_g_ema
        if self.mc_K <= 1:
            pl, pt = net(real, finetune=True)
            j = pl[0].clamp(0, 1)
            t = pt[0]
            return j, t, torch.zeros_like(j), torch.zeros_like(t)

        enable_mc_dropout(net)
        try:
            js, ts = [], []
            for _ in range(self.mc_K):
                pl, pt = net(real, finetune=True)
                js.append(pl[0].clamp(0, 1))
                ts.append(pt[0])
        finally:
            # Always restore determinism, even if a forward pass raises.
            disable_mc_dropout(net)

        j_stack = torch.stack(js, dim=0)
        t_stack = torch.stack(ts, dim=0)
        return (j_stack.mean(0), t_stack.mean(0),
                j_stack.var(0, unbiased=False), t_stack.var(0, unbiased=False))

    def init_clip(self):
        clip_model_type = self.opt['colabator'].get('clip_model_type', None)
        checkpoint = self.opt['colabator'].get('pretrained_clip_weight', None)
        tokenizer_type = self.opt['colabator'].get('tokenizer_type', None)
        self.clip_better = self.opt['colabator'].get('clip_better', None)
        self.degradation_type = self.opt['colabator'].get('degradation_type', None)
        self.weight_map_calculation = self.opt['colabator'].get('weight_map_calculation', 'addition')

        self.clip_model, self.clip_preprocess = open_clip.create_model_from_pretrained(clip_model_type,                                                               pretrained=checkpoint)
        self.clip_model = self.model_to_device(self.clip_model)
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer(tokenizer_type)
        degradations = ['motion-blurry', 'hazy', 'jpeg-compressed', 'low-light', 'noisy', 'raindrop', 'rainy',
                        'shadowed', 'snowy', 'uncompleted']
        text = self.tokenizer(degradations)
        text = text.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            if self.opt['dist']:
                text_features = self.clip_model.module.encode_text(text)
            else:
                text_features = self.clip_model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            self.text_features = text_features

    def init_nriqa(self):
        nr_iqa_type = self.opt['colabator'].get('nr_iqa_type', None)
        self.nr_iqa_better = self.opt['colabator'].get('nr_iqa_better', None)
        self.nr_iqa_scale = self.opt['colabator'].get('nr_iqa_scale', None)
        self.nr_iqa = pyiqa.create_metric(nr_iqa_type)
        self.nr_iqa = self.model_to_device(self.nr_iqa).eval()

    def init_mmb(self):
        self.memory_bank = memory_bank.Memory_bank().to('cpu')
        memory_bank_path = self.opt['path'].get('pretrain_network_memory_bank', None)
        if memory_bank_path is not None:
            self.memory_bank.load_state_dict(torch.load(memory_bank_path))

    def block_image(self, image, block_size):
        B, C, H, W = image.size()
        BH, BW = block_size

        # Calculate the image shape after the block
        num_H = H // BH
        num_W = W // BW

        # Reshape the image into a block shape
        blocked_image = image.view(B, C, num_H, BH, num_W, BW)

        # Exchange dimensions so that the blocks are in the right place.
        blocked_image = blocked_image.permute(2, 4, 0, 1, 3, 5).contiguous()

        # Reshaped to the original shape
        blocked_image = blocked_image.view(num_H * num_W * B, C, BH, BW)

        return blocked_image

    def unblock_image(self, blocked_image, block_size, original_shape):
        B, C, H, W = original_shape
        BH, BW = block_size

        # Calculate the image shape after the block
        num_H = H // BH
        num_W = W // BW

        # Reshape the image into a block shape
        blocked_image = blocked_image.view(num_H, num_W, B, 1, 1)

        # Exchange dimensions so that the blocks are in the right place.
        blocked_image = blocked_image.permute(2, 0, 3, 1, 4).contiguous()

        # Reshaped to the original shape
        blocked_image = blocked_image.view(B, 1, num_H, num_W)

        # Resize to original shape
        blocked_image = torch.nn.functional.interpolate(blocked_image, (H, W), mode='bilinear', align_corners=False)

        return blocked_image

    def get_clip_degrad_rate(self, img):
        image = self.clip_preprocess(img)
        sum_probs = 0
        for degradation in self.degradation_type:
            with torch.no_grad(), torch.cuda.amp.autocast():
                if self.opt['dist']:
                    _, degra_features = self.clip_model.module.encode_image(image, control=True)
                else:
                    _, degra_features = self.clip_model.encode_image(image, control=True)
                # image_features /= image_features.norm(dim=-1, keepdim=True)
                degra_features /= degra_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * degra_features @ self.text_features.T).softmax(dim=-1)
                probs = text_probs[:, degradation]
                sum_probs = sum_probs + probs
        return sum_probs

    def get_batch_avg_degrad_rate(self, imgs):
        sum_rate = self.get_clip_degrad_rate(imgs)
        sum_rate = sum_rate.mean()
        return sum_rate / imgs.shape[0]


    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'real' in data:
            self.real = data['real'].to(self.device)
        if 't' in data:
            self.transmission = data['t'].to(self.device)
        if 'real_strong' in data:
            self.real_strong = data['real_strong'].to(self.device)
        if 'real_name' in data:
            self.real_name = data['real_name']
        if 'mini_gt_size' in data:
            self.mini_gt_size = data['mini_gt_size']
        if 'gt_size' in data:
            self.gt_size = data['gt_size']

        if self.is_train and self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def test(self):
        window_size = self.opt['val'].get('window_size', 0)
        if window_size:
            lq, mod_pad_h, mod_pad_w = self.pad_test(self.lq,window_size)
        else:
            lq = self.lq

        # if hasattr(self, 'net_g_ema'):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.outputs, self.output_transmissions, self.step_images, self.recon_images = self.net_g_ema(
                    img=lq, debug=True)
                self.output = self.outputs[0].clamp(0,1)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.outputs, self.output_transmissions, self.step_images, self.recon_images = self.net_g(
                    img=lq, debug=True)
                self.output = self.outputs[0].clamp(0,1)
            self.net_g.train()


        if window_size:
            scale = self.opt.get('scale', 1)
            _, _, h, w = self.output.size()
            self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def labal_selection(self, teacher_transmission, teacher, joint_uncertainty_map=None): # teacher is pseudo label
        teacher_tar = teacher.detach()
        original_shape = teacher_tar.size()
        with torch.no_grad():
            # block image
            teacher_tar_blocks = self.block_image(teacher_tar, (self.block_size, self.block_size))
            if self.opt['colabator'].get('use_nr_iqa', False):
                # local
                teacher_nr_iqa_score_sequence = self.nr_iqa(teacher_tar_blocks)
                # global
                teacher_nr_iqa_score = (self.nr_iqa(teacher_tar) - self.nr_iqa_scale[0]) / (
                            self.nr_iqa_scale[1] - self.nr_iqa_scale[0])
                # unblock image
                if self.nr_iqa_scale != 'sigmoid':
                    teacher_nr_iqa_score_mask = (self.unblock_image(teacher_nr_iqa_score_sequence,
                                                                    (self.block_size, self.block_size),
                                                                    original_shape) - self.nr_iqa_scale[0]) / (
                                                            self.nr_iqa_scale[1] - self.nr_iqa_scale[0])
                else:
                    teacher_nr_iqa_score_mask = torch.sigmoid(self.unblock_image(teacher_nr_iqa_score_sequence,
                                                                                 (self.block_size, self.block_size),
                                                                                 original_shape))
                if self.nr_iqa_better == 'higher': # if higher is better, then the score is higher, the mask is 1
                    teacher_nr_iqa_score_mask = teacher_nr_iqa_score_mask
                    teacher_nr_iqa_score = teacher_nr_iqa_score
                else: # if lower is better, then the score is lower, the mask is 1
                    teacher_nr_iqa_score_mask = 1 - teacher_nr_iqa_score_mask
                    teacher_nr_iqa_score = 1 - teacher_nr_iqa_score
            else:
                teacher_nr_iqa_score_mask = 0
                teacher_nr_iqa_score = 0

            if self.opt['colabator'].get('use_clip', False):
                # local
                teacher_score_sequence = self.get_clip_degrad_rate(teacher_tar_blocks)
                # global
                teacher_score = self.get_clip_degrad_rate(teacher_tar)
                # unblock image
                teacher_score_mask = len(self.degradation_type) - self.unblock_image(teacher_score_sequence,
                                                                                     (self.block_size, self.block_size),
                                                                                     original_shape)
                if self.clip_better == 'higher':
                    teacher_score = teacher_score
                    teacher_score_mask = teacher_score_mask
                else:
                    teacher_score = len(self.degradation_type) - teacher_score
                    teacher_score_mask = len(self.degradation_type) - teacher_score_mask
            else:
                teacher_score_mask = 0
                teacher_score = 0

        if joint_uncertainty_map is not None and self.quadtree_router is not None:
            # Joint quadtree spatial routing. The router scores each leaf itself
            # with the same DA-CLIP + MUSIQ evaluator used above, so its output
            # already is the multi-scale reliability mask M and replaces the
            # uniform-grid mask rather than multiplying into it.
            teacher_mask = self.quadtree_router(teacher_tar, joint_uncertainty_map)
            # Retention here is the fraction of the image the router did not
            # hard-zero, i.e. the area still able to contribute gradient.
            self._data_retention_rate = (teacher_mask > 0).float().mean().item()

        elif joint_uncertainty_map is not None:
            # Fallback: flat ASM gate on the uniform block grid, used when
            # colabator.use_quadtree is false.
            uncertainty_blocks = self.block_image(joint_uncertainty_map, (self.block_size, self.block_size))
            block_uncertainties = uncertainty_blocks.mean(dim=(1, 2, 3))
            confidence_gate = (block_uncertainties < self.tau_asm).float()
            confidence_gate_mask = self.unblock_image(confidence_gate, (self.block_size, self.block_size), original_shape)

            if self.weight_map_calculation == 'multiplication':
                teacher_mask = teacher_nr_iqa_score_mask * (teacher_score_mask / len(self.degradation_type)) * confidence_gate_mask
            else:
                teacher_mask = ((teacher_nr_iqa_score_mask + teacher_score_mask) / (len(self.degradation_type) + 1)) * confidence_gate_mask
            self._data_retention_rate = confidence_gate.mean().item()

        else:
            # No uncertainty signal at all: stock Colabator weighting.
            if self.weight_map_calculation == 'multiplication':
                teacher_mask = teacher_nr_iqa_score_mask * (teacher_score_mask / len(self.degradation_type))
            else:
                teacher_mask = (teacher_nr_iqa_score_mask + teacher_score_mask) / (len(self.degradation_type) + 1)
            self._data_retention_rate = 1.0

        teacher, teacher_transmission, teacher_nr_iqa_score, teacher_score, teacher_mask = self.memory_bank(self.real_name, teacher, teacher_transmission, teacher_nr_iqa_score, teacher_score, self.device, teacher_mask)
        teacher = teacher.to(self.device)
        teacher_transmission = teacher_transmission.to(self.device)

        return teacher_transmission, teacher, teacher_mask



    def optimize_parameters(self, current_iter, log_vars=None):
        if current_iter % self.opt['logger']['save_checkpoint_freq'] == 0:
            self.save_memory_bank(current_iter)

        self.optimizer_g.zero_grad()
        outputs, transmissions = self.net_g(self.lq, finetune=True)
        output = outputs[0].clamp(0, 1)
        transmission = transmissions[0]
        recon_lq = output * transmission + (1 - transmission)

        # Teacher-only MC Dropout: K stochastic passes give the means that feed
        # the ASM reconstruction and the variances that feed the router
        # (paper Eq. 6-7).
        pseudo_label, pseudo_transmission, var_j, var_t = self.mc_teacher_forward(self.real)

        # Physical drift error (paper Eq. 4-5): how well the teacher's own
        # prediction re-explains the observed hazy image under the ASM.
        I_recon = pseudo_label * pseudo_transmission + (1 - pseudo_transmission)
        phys_error = torch.mean((I_recon - self.real) ** 2, dim=1, keepdim=True)

        # Joint uncertainty map (paper Eq. 8). With the quadtree enabled this
        # fuses epistemic variance with physical inconsistency; with it disabled
        # the flat ASM gate thresholds the raw drift error directly, so the two
        # paths stay on their own natural scales.
        if self.use_quadtree:
            joint_uncertainty_map = joint_uncertainty(var_j, var_t, phys_error)
        else:
            joint_uncertainty_map = phys_error

        real_outputs, real_transmissions = self.net_g(self.real_strong, finetune=True)
        real_output = real_outputs[0]
        real_transmission = real_transmissions[0]
        pseudo_transmission, pseudo_label, pseudo_mask = self.labal_selection(pseudo_transmission, pseudo_label, joint_uncertainty_map)
        recon_real_lq = real_output * real_transmission + (1 - real_transmission)

        ###########################################
        # this is a sample
        # you can modify this part for your losses

        l_total = 0
        loss_dict = OrderedDict()

        if self.cri_pix:
            l_pix = self.cri_pix(output, self.gt).mean()
            if pseudo_label is not None:
                l_pix += (self.cri_pix(real_output, pseudo_label) * pseudo_mask * 2).mean()
            loss_dict['l_pix'] = l_pix
            l_pix = l_pix * 5
            l_total += l_pix

            l_asm = (self.cri_pix(recon_real_lq, self.real_strong) * pseudo_mask).mean() * 0.01
            loss_dict['l_asm'] = l_asm
            l_total += l_asm

        if self.opt['train'].get('use_clip_loss', False):
            clip_loss = self.get_batch_avg_degrad_rate(output)
            clip_loss += self.get_batch_avg_degrad_rate(real_output)
            loss_dict['clip_loss'] = clip_loss
            l_total += clip_loss

        # perceptual loss
        if self.cri_contrastperceptual:
            l_percep, l_style = self.cri_contrastperceptual.standard_perceptual_loss(output, self.gt)
            l_contrast_percep, l_contrast_style = self.cri_contrastperceptual(real_output, pseudo_label, self.real_strong)
            if l_percep is not None:
                l_total += (l_percep * 0.2)
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += (l_style * 0.2)
                loss_dict['l_style'] = l_style
            if l_contrast_percep is not None:
                l_total += l_contrast_percep * pseudo_mask.mean()
                loss_dict['l_contrast_percep'] = l_contrast_percep * pseudo_mask.mean()
            if l_contrast_style is not None:
                l_total += l_contrast_style * pseudo_mask.mean()
                loss_dict['l_contrast_style'] = l_contrast_style * pseudo_mask.mean()

        ###########################################

        l_total.backward()
        self.optimizer_g.step()

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        # Log data retention rate from Quadtree gating
        if hasattr(self, '_data_retention_rate'):
            loss_dict['data_retention'] = torch.tensor(self._data_retention_rate)

        self.log_dict = self.reduce_loss_dict(loss_dict)
