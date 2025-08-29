import torch
import torch.nn as nn
from torch.nn import functional as F
try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

def gather_features(image_features, text_features, local_loss=False, gather_with_grad=False, rank=0, world_size=1, use_horovod=False):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    elif gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
    return (all_image_features, all_text_features)

class ClipLoss(nn.Module):

    def __init__(self, local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1, use_horovod=False):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            (all_image_features, all_text_features) = gather_features(image_features, text_features, self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        return (logits_per_image, logits_per_text)

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        (logits_per_image, logits_per_text) = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return {'contrastive_loss': total_loss} if output_dict else total_loss

def gather_features_da(image_features, text_features, valid_caption_mask, local_loss=False, gather_with_grad=False, rank=0, world_size=1, use_horovod=False):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    elif gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        all_valid_caption_mask = torch.cat(torch.distributed.nn.all_gather(valid_caption_mask), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        gathered_valid_caption_mask = [torch.zeros_like(valid_caption_mask) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        dist.all_gather(gathered_valid_caption_mask, valid_caption_mask)
        if not local_loss:
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
            gathered_valid_caption_mask[rank] = valid_caption_mask
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
        all_valid_caption_mask = torch.cat(gathered_valid_caption_mask, dim=0)
    return (all_image_features, all_text_features, all_valid_caption_mask)

class Clip_DALoss(nn.Module):

    def __init__(self, local_loss=False, gather_with_grad=False, cache_labels=False, rank=0, world_size=1, use_horovod=False, mar_loss=False, neg_loss=False, hardnegative=False, neg_loss_weight=0, mar_loss_weight=0.2, threshold_type='mean', positive_margin_loss=False, positive_margin_loss_weight=0.2):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.prev_num_logits = 0
        self.labels = {}
        self.mar_loss = mar_loss
        self.neg_loss = neg_loss
        self.neg_loss_weight = neg_loss_weight
        self.mar_loss_weight = mar_loss_weight
        self.threshold_type = threshold_type
        self.hardnegative = hardnegative
        self.positive_margin_loss = positive_margin_loss
        self.positive_margin_loss_weight = positive_margin_loss_weight
        if self.mar_loss and self.positive_margin_loss:
            self.alpha = nn.Parameter(torch.randn(1))

    def forward(self, image_features, text_features, valid_caption_mask, logit_scale, thresholds):
        device = image_features.device
        (mar_loss, neg_loss) = (0.0, 0.0)
        gt_similarity_diag = None
        if self.world_size > 1:
            (all_image_features, all_text_features, all_valid_caption_mask) = gather_features_da(image_features, text_features, valid_caption_mask, self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            num_negatives = 5
            caption_types = torch.tensor(([1] * image_features.shape[0] + [2] * image_features.shape[0] * num_negatives) * self.world_size)
            gt_all_text_features = all_text_features[caption_types == 1]
            da_all_text_features = all_text_features[caption_types == 2]
            (gt_len, feature_size) = (all_image_features.shape[0], all_image_features.shape[-1])
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                if self.hardnegative:
                    all_text_features = torch.cat([gt_all_text_features, da_all_text_features])
                    logits_per_image = logit_scale * all_image_features @ all_text_features.T
                else:
                    logits_per_image = logit_scale * all_image_features @ gt_all_text_features.T
                logits_per_text = logit_scale * gt_all_text_features @ all_image_features.T
                if self.mar_loss:
                    da_logits_per_image = logit_scale * (da_all_text_features.reshape(gt_len, -1, feature_size) @ all_image_features.unsqueeze(-1)).squeeze() * all_valid_caption_mask
                    (mar_loss, thresholds) = self.get_mar_loss(logits_per_image, da_logits_per_image, all_valid_caption_mask, thresholds)
                if self.neg_loss:
                    text_embedding_matrix = logit_scale * gt_all_text_features @ da_all_text_features.T
                    neg_loss += self.get_neg_loss(logits_per_image, text_embedding_matrix)
        else:
            (gt_len, feature_size) = (image_features.shape[0], image_features.shape[-1])
            gt_text_features = text_features[:gt_len]
            da_text_features = text_features[gt_len:]
            base_logits_per_image = logit_scale * image_features @ gt_text_features.T
            logits_per_text = base_logits_per_image.T
            if self.hardnegative:
                all_text_features = torch.cat([gt_text_features, da_text_features])
                logits_per_image = logit_scale * image_features @ all_text_features.T
            else:
                logits_per_image = base_logits_per_image
            gt_similarity_diag = base_logits_per_image.diag()
            if self.mar_loss:
                da_logits_per_image = logit_scale * (da_text_features.reshape(gt_len, -1, feature_size) @ image_features.unsqueeze(-1)).squeeze() * valid_caption_mask
                (mar_loss_pos, mar_loss_neg, thresholds) = self.get_mar_loss(gt_similarity_diag, da_logits_per_image, valid_caption_mask, thresholds)
            if self.neg_loss:
                da_text_features_reshaped = da_text_features.reshape(gt_len, -1, feature_size)
                gt_text_features_expanded = gt_text_features.unsqueeze(1)
                semantic_diff_vectors = da_text_features_reshaped - gt_text_features_expanded
                text_distances = torch.norm(semantic_diff_vectors, p=2, dim=-1)
                semantic_diff_norm = F.normalize(semantic_diff_vectors, p=2, dim=-1, eps=1e-08)
                distances_detached = text_distances.detach()
                direction_detached = semantic_diff_norm.detach()
                final_perturbation = distances_detached.unsqueeze(-1) * direction_detached
                image_features_expanded = image_features.unsqueeze(1)
                image_negative_features = image_features_expanded + final_perturbation
                neg_loss = self.get_neg_loss(logit_scale, image_features, image_negative_features, gt_text_features, da_text_features)
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        total_loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        if self.mar_loss and 'mar_loss_pos' in locals():
            mar_loss = mar_loss_pos + mar_loss_neg
            total_loss += mar_loss * self.mar_loss_weight
        if self.neg_loss and 'neg_loss' in locals():
            total_loss += neg_loss * self.neg_loss_weight
        if 'mar_loss_combined' in locals():
            mar_loss = mar_loss_combined
        elif 'mar_loss' not in locals():
            mar_loss = torch.tensor(0.0)
        if 'neg_loss' not in locals():
            neg_loss = torch.tensor(0.0)
        return (total_loss, thresholds, mar_loss, neg_loss)

    def get_mar_loss(self, gt_similarity_diag: torch.Tensor, da_logits_per_image: torch.Tensor, valid_caption_mask, thresholds: torch.Tensor):
        gt_similarity = gt_similarity_diag.reshape(-1, 1).expand_as(da_logits_per_image)
        mar_loss_neg = nn.functional.relu(thresholds + da_logits_per_image - gt_similarity) * valid_caption_mask
        mar_loss_neg = mar_loss_neg.mean()
        mar_loss_pos = torch.tensor(0.0, device=gt_similarity_diag.device)
        if self.positive_margin_loss:
            if hasattr(self, 'alpha'):
                self.alpha.data.clamp_(min=0.2)
                mar_loss_pos = F.relu(self.alpha - gt_similarity_diag).mean()
            else:
                pass
        if self.threshold_type == 'mean':
            mask = da_logits_per_image != 0
            valid_counts = mask.sum(dim=0)
            average_similarity_for_types = torch.where(valid_counts > 0, (da_logits_per_image * mask).sum(dim=0) / valid_counts, torch.zeros_like(valid_counts, dtype=da_logits_per_image.dtype))
            thresholds = (gt_similarity.mean(0) - average_similarity_for_types).expand_as(gt_similarity)
            thresholds = thresholds.detach()
        elif self.threshold_type == 'max':
            (thresholds, max_indices) = (gt_similarity * valid_caption_mask - da_logits_per_image).max(0)
            thresholds = thresholds.expand_as(gt_similarity) / 5
            thresholds = thresholds.detach()
        return (mar_loss_pos, mar_loss_neg, thresholds)

    def get_neg_loss(self, logit_scale: torch.Tensor, image_positive_features: torch.Tensor, image_negative_features: torch.Tensor, text_positive_features: torch.Tensor, text_negative_features: torch.Tensor):
        device = image_positive_features.device
        text_logits = logit_scale * text_positive_features @ text_negative_features.T
        image_neg_flat = image_negative_features.reshape(-1, image_negative_features.shape[-1])
        image_logits = logit_scale * image_positive_features @ image_neg_flat.T
        labels = torch.arange(text_logits.shape[0], device=device, dtype=torch.long)
        loss_text_side = F.cross_entropy(text_logits, labels)
        loss_image_side = F.cross_entropy(image_logits, labels)
        return loss_text_side + loss_image_side