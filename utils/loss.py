import torch
from torch.nn import CrossEntropyLoss


def loss_func(outputs, labels, loss_mask):

    lm_logits = outputs["logits"].contiguous()
    labels = labels.to(device=lm_logits.device).contiguous()
    loss_mask = loss_mask.to(device=lm_logits.device)
    # logits: (bs, l, v); labels, loss_mask: (bs, l)

    # lm loss
    bsz = labels.shape[0]
    loss_func = CrossEntropyLoss(reduction='none')
    losses = loss_func(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))  # logits: (bs * l, v); labels: (bs * l,)
    # losses -> (bs, l)
    losses = losses.contiguous().view(bsz, -1)

    loss_mask = loss_mask.view(-1)
    losses = losses.view(-1)
    loss_lm = torch.sum(losses * loss_mask) / loss_mask.sum()

    return loss_lm
