# contrastive_loss.py
import torch
import torch.nn.functional as F


class ContrastiveLoss:
    def __init__(self, embedding_layer, ignore_occlusion_label=None,
                 inner_threshold_on=False, inner_occlusion_label=None,
                 inner_threshold=0.6, outer_threshold=0.4, device='cpu'):
        self.embedding_layer = embedding_layer
        self.ignore_occlusion_label = ignore_occlusion_label
        self.inner_threshold_on = inner_threshold_on
        self.inner_occlusion_label = inner_occlusion_label
        self.inner_threshold = inner_threshold
        self.outer_threshold = outer_threshold
        self.device = device

    def compute_loss(self, features, labels):
        feature = features[self.embedding_layer]

        if self.ignore_occlusion_label is not None:
            ignore_label = torch.tensor(self.ignore_occlusion_label, device=self.device)
            indices_to_keep = (labels != ignore_label).nonzero(as_tuple=True)[0]
            feature = feature[indices_to_keep]
            labels = labels[indices_to_keep]

        if self.inner_threshold_on:
            is_inner_label = (labels == self.inner_occlusion_label)
            inner_label_row = is_inner_label.unsqueeze(0)
            inner_label_col = is_inner_label.unsqueeze(1)
            inner_label_matrix = (inner_label_row | inner_label_col).float()

        if labels.numel() == 0:
            return torch.zeros(1, device=self.device)

        batch_size, _, _, _ = feature.size()
        feature = F.adaptive_avg_pool2d(feature, (1, 1))
        feature = feature.view(batch_size, -1)
        feature = F.normalize(feature)
        cos_matrix = feature.mm(feature.t())
        pos_label_matrix = torch.stack([labels == labels[i] for i in range(batch_size)]).float()
        neg_label_matrix = 1 - pos_label_matrix
        pos_cos_matrix = 1 - cos_matrix
        neg_cos_matrix = cos_matrix - inner_label_matrix * self.inner_threshold - (1 - inner_label_matrix) * self.outer_threshold
        neg_cos_matrix[neg_cos_matrix < 0] = 0
        con_loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
        con_loss /= (batch_size * batch_size)

        return con_loss