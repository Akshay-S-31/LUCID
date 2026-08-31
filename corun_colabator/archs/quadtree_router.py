"""Joint quadtree spatial routing for LUCID.

Implements Sec. "Joint Quadtree Spatial Routing" of the LUCID paper. The router
turns three per-pixel unreliability signals into a single multi-scale
reliability mask ``M in [0,1]^(H x W)`` that gates the semi-supervised losses.

Joint uncertainty (paper Eq. 8)::

    U_joint = 1/3 * sigma^2_J_hat + 1/3 * sigma^2_T_hat + 1/3 * eps_phys_hat

where each map is min-max normalised to [0,1] independently, so the three
signals are individually calibrated before being combined with equal weights.
``sigma^2_J`` and ``sigma^2_T`` are the teacher-side MC-Dropout variances on the
scene and transmission estimates (epistemic uncertainty); ``eps_phys`` is the
physical drift error of the ASM reconstruction (physical inconsistency).

Quadtree splitter: the image is partitioned into non-overlapping ``64x64`` root
blocks. A block whose mean joint uncertainty is at or below ``tau_q`` is
confidently reliable and is scored once as a leaf. An uncertain block is split
into four children and each child is re-examined at the next level down, through
``64 -> 32 -> 16 -> 8``. Blocks at the finest level stop splitting and become
leaves regardless. Any leaf whose mean joint uncertainty exceeds ``tau_crit`` is
deemed irrecoverable and is hard-zeroed.

Each surviving leaf is scored by the same combined DA-CLIP haze-density and
MUSIQ aesthetic evaluator that Colabator already uses, at the leaf's own spatial
resolution, and the resulting per-leaf weight is painted into the output mask.

The router is inference-only: it runs under ``torch.no_grad()`` and contributes
no gradient of its own. It only decides how strongly each region of a
pseudo-label is allowed to contribute.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['QuadtreeRouter', 'minmax_normalise', 'joint_uncertainty']


def minmax_normalise(x, eps=1e-8):
    """Per-sample min-max normalisation of a [B, 1, H, W] map to [0, 1].

    Normalising per sample rather than per batch keeps the router's decisions
    independent across images, so one pathological image cannot rescale the
    uncertainties of its batch-mates. A constant map normalises to all-zeros,
    which reads as "uniformly certain" -- the correct reading, since a constant
    map carries no information about which region is worse than another.
    """
    B = x.shape[0]
    flat = x.reshape(B, -1)
    lo = flat.min(dim=1).values.view(B, 1, 1, 1)
    hi = flat.max(dim=1).values.view(B, 1, 1, 1)
    return (x - lo) / (hi - lo + eps)


def joint_uncertainty(var_j, var_t, phys_err, weights=(1 / 3., 1 / 3., 1 / 3.)):
    """Fuse the three unreliability signals into U_joint (paper Eq. 8).

    Args:
        var_j (Tensor): teacher MC-Dropout variance on the scene, [B, C, H, W]
            or [B, 1, H, W]. Averaged over channels if needed.
        var_t (Tensor): teacher MC-Dropout variance on the transmission.
        phys_err (Tensor): ASM physical drift error, [B, 1, H, W].
        weights (tuple): (alpha, beta, gamma). Defaults to equal thirds.

    Returns:
        Tensor: U_joint in [0, 1], shape [B, 1, H, W].
    """
    def _to_map(v):
        if v.shape[1] != 1:
            v = v.mean(dim=1, keepdim=True)
        return minmax_normalise(v)

    a, b, g = weights
    return a * _to_map(var_j) + b * _to_map(var_t) + g * _to_map(phys_err)


class QuadtreeRouter(nn.Module):
    """Recursive quadtree splitter over the joint uncertainty map.

    Args:
        scorer (callable): maps a batch of image crops [N, 3, h, w] to a scalar
            reliability weight per crop, [N], in [0, 1]. In LUCID this is the
            combined DA-CLIP + MUSIQ evaluator; see
            ``Colabator_by_Depth.build_block_scorer``.
        sizes (sequence[int]): block sizes from root to finest, e.g.
            (64, 32, 16, 8). Must be descending and each a multiple of the next.
        tau_q (float): split threshold on mean U_joint. At or below this the
            block is confidently reliable and becomes a leaf.
        tau_crit (float): a leaf whose mean U_joint exceeds this is discarded
            (weight forced to zero).
    """

    def __init__(self, scorer, sizes=(64, 32, 16, 8), tau_q=0.3, tau_crit=0.7):
        super(QuadtreeRouter, self).__init__()
        sizes = tuple(int(s) for s in sizes)
        if len(sizes) < 1:
            raise ValueError('sizes must contain at least one block size')
        for coarse, fine in zip(sizes, sizes[1:]):
            if fine >= coarse:
                raise ValueError(f'sizes must strictly descend, got {sizes}')
            if coarse % fine != 0:
                raise ValueError(
                    f'each size must divide the previous one, got {coarse} -> {fine}')
        self.scorer = scorer
        self.sizes = sizes
        self.tau_q = float(tau_q)
        self.tau_crit = float(tau_crit)

    @torch.no_grad()
    def forward(self, image, u_joint):
        """Build the reliability mask M for one batch.

        Args:
            image (Tensor): the pseudo-label being judged, [B, 3, H, W] in [0,1].
            u_joint (Tensor): joint uncertainty in [0,1], [B, 1, H, W].

        Returns:
            Tensor: mask M in [0, 1], shape [B, 1, H, W], differentiably
            broadcastable against a per-pixel loss.
        """
        B, _, H, W = image.shape
        mask = image.new_zeros((B, 1, H, W))

        # Regions are tracked as explicit (batch index, top, left, nominal size)
        # tuples rather than by recursing on tensors, so that every leaf at a
        # given depth can be scored together in one batched call. The root grid
        # covers the full image: when H or W is not a multiple of the root size
        # the trailing blocks are partial, and their extent is clamped wherever
        # they are read or painted. No pixel is left unscored.
        root = self.sizes[0]
        pending = [(b, top, left, root)
                   for b in range(B)
                   for top in range(0, H, root)
                   for left in range(0, W, root)]

        leaves = []
        for level, size in enumerate(self.sizes):
            if not pending:
                break
            is_finest = (level == len(self.sizes) - 1)
            nxt = []
            for (b, top, left, sz) in pending:
                bot, right = min(top + sz, H), min(left + sz, W)
                if bot <= top or right <= left:
                    continue
                block_u = u_joint[b, :, top:bot, left:right].mean()
                if block_u <= self.tau_q or is_finest:
                    leaves.append((b, top, left, sz, float(block_u)))
                else:
                    child = self.sizes[level + 1]
                    for dy in range(0, sz, child):
                        for dx in range(0, sz, child):
                            if top + dy < H and left + dx < W:
                                nxt.append((b, top + dy, left + dx, child))
            pending = nxt

        if not leaves:
            return mask

        self._paint(image, mask, leaves)
        return mask

    def _paint(self, image, mask, leaves):
        """Score the leaves and write their weights into the mask.

        Leaves are grouped by their block size and scored one group at a time,
        so each region is evaluated at its own spatial resolution -- a block that
        was split down to 8x8 is judged at 8x8, and one that stayed at 64x64 is
        judged at 64x64. This is what the paper means by scoring children "at
        higher spatial resolution"; resampling every leaf to a common size would
        throw away exactly the detail the MUSIQ and DA-CLIP evaluators read.
        """
        H, W = image.shape[2], image.shape[3]

        by_size = {}
        for (b, top, left, sz, block_u) in leaves:
            bot, right = min(top + sz, H), min(left + sz, W)
            if bot <= top or right <= left:
                continue
            by_size.setdefault(sz, []).append((b, top, bot, left, right, block_u))

        for sz, group in by_size.items():
            crops = []
            for (b, top, bot, left, right, _) in group:
                crop = image[b:b + 1, :, top:bot, left:right]
                # Only partial edge blocks are resampled, and only up to their
                # own nominal size, so the scorer sees a uniform batch.
                if crop.shape[-2:] != (sz, sz):
                    crop = F.interpolate(crop, size=(sz, sz),
                                         mode='bilinear', align_corners=False)
                crops.append(crop)

            scores = self.scorer(torch.cat(crops, dim=0))
            scores = scores.reshape(-1).to(mask.dtype).clamp(0, 1)

            for idx, (b, top, bot, left, right, block_u) in enumerate(group):
                if block_u > self.tau_crit:
                    # Irrecoverable region: hard zero, no gradient at all.
                    mask[b, :, top:bot, left:right] = 0.0
                else:
                    mask[b, :, top:bot, left:right] = scores[idx]
