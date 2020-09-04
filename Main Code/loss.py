import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

GAMMA1 = 4.0
GAMMA3 = 10.0
GAMMA2 = 5.0

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim.
    """
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def words_loss(img_features_, words_emb, labels,
               cap_lens=None, class_ids=None, batch_size=None):
    """
        words_emb(query): batch x nef x seq_len
        img_features(context): batch x nef x 17 x 17
    """
    batch_size = len(img_features_)
    masks = []
    att_maps = []
    similarities = []

    # sketch_F = []
    # for img in img_features:
    #     sketch_F.append(F.adaptive_max_pool2d(img, (1, 1)).squeeze(-1).squeeze(-1))
    # img_features = torch.stack(sketch_F)

    try:
        img_features = torch.stack([F.adaptive_max_pool2d(img, (1, 1)).squeeze(-1).squeeze(-1) for img in img_features_])
    except RuntimeError:
        for img in img_features:
            print(F.adaptive_max_pool2d(img, (1, 1)).squeeze(-1).squeeze(-1).shape)


    #Batch, 512, 16
    for i in range(batch_size):
        # if class_ids is not None:
        #     mask = (class_ids == class_ids[i]).astype(np.uint8)
        #     mask[i] = 0
        #     masks.append(mask.reshape((1, -1)))
        # Get the i-th text description
        #words_num = cap_lens[i]
        #word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()

        word = torch.stack([F.adaptive_max_pool2d(word, (1, 1)).view(512) for word in words_emb[i]]).unsqueeze(0)
        word = word.repeat(batch_size, 1, 1).permute([0,2,1])
        context = img_features.permute([0,2,1])

        weiContext, attn = func_attention(word, context, GAMMA1)
        # weiContext: torch.Size([32, 512, 10])
        # word: torch.Size([32, 512, 10])

        #att_maps.append(attn[i].unsqueeze(0).contiguous())
        word = word.transpose(1, 2).contiguous()
        weiContext = weiContext.transpose(1, 2).contiguous()
        words_num = word.size(1)
        word = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)
        #
        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)

        row_sim.mul_(GAMMA2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)
    if class_ids is not None:
        masks = np.concatenate(masks, 0)
        masks = torch.ByteTensor(masks)
        masks = masks.cuda()

    similarities = similarities * GAMMA3

    if class_ids is not None:
        similarities.data.masked_fill_(masks, -float('inf'))
    similarities1 = similarities.transpose(0, 1)

    if labels is not None:
        loss0 = nn.CrossEntropyLoss()(similarities, labels.cuda())
        loss1 = nn.CrossEntropyLoss()(similarities1, labels.cuda())
    else:
        loss0, loss1 = None, None
    return loss0, loss1 #, att_maps


def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    #ih, iw = context.size(2), context.size(3)
    sourceL = context.size(2)

    # --> batch x sourceL x ndf
    #context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size*sourceL, queryL)
    attn = nn.Softmax()(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)

    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, sourceL)


def masked_softmax(logits, mask=None):
    eps = 1e-20
    probs = F.softmax(logits, dim=1)
    if mask is not None:
        mask = mask.float()
        probs = probs * mask + eps
        probs = probs / probs.sum(1, keepdim=True)
    return probs

def st_gumbel_softmax(logits, temperature=1.0, mask=None):
    """
    Return the result of Straight-Through Gumbel-Softmax Estimation.
    It approximates the discrete sampling via Gumbel-Softmax trick
    and applies the biased ST estimator.
    In the forward propagation, it emits the discrete one-hot result,
    and in the backward propagation it approximates the categorical
    distribution via smooth Gumbel-Softmax distribution.

    Args:
        logits (tensor): A un-normalized probability values,
            which has the size (batch_size, num_classes)
        temperature (float): A temperature parameter. The higher
            the value is, the smoother the distribution is.
        mask (tensor, optional): If given, it masks the softmax
            so that indices of '0' mask values are not selected.
            The size is (batch_size, num_classes).

    Returns:
        y: The sampled output, which has the property explained above.
    """

    eps = 1e-20
    u = logits.data.new(*logits.size()).uniform_()
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    y = masked_softmax(logits=y / temperature, mask=mask)
    y_argmax = y.max(1)[1]
    y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y


def convert_to_one_hot(indices, num_classes):
    """
    Args:
        indices (tensor): A vector containing indices,
            whose size is (batch_size,).
        num_classes (tensor): The number of classes, which would be
            the second dimension of the resulting one-hot matrix.

    Returns:
        result: The one-hot matrix of size (batch_size, num_classes).
    """

    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.new_zeros(batch_size, num_classes).scatter_(1, indices, 1)
    return one_hot


def greedy_select(logits, mask=None):
    probs = masked_softmax(logits=logits, mask=mask)
    one_hot = convert_to_one_hot(indices=probs.max(1)[1],
                                 num_classes=logits.size(1))
    return one_hot