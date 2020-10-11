from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

# from .deterministic_alignment import compute_deterministic_alignment_loss
from .stochastic_alignment import compute_stochastic_alignment_loss


def compute_alignment_loss(embs,
                           batch_size,
                           steps=None,
                           seq_lens=None,
                           stochastic_matching=False,
                           normalize_embeddings=False,
                           loss_type='classification',
                           similarity_type='l2',
                           num_cycles=20,
                           cycle_length=2,
                           temperature=0.1,
                           label_smoothing=0.1,
                           variance_lambda=0.001,
                           huber_delta=0.1,
                           normalize_indices=True):

    # Get the number of timestemps in the sequence embeddings.
    num_steps = embs.size(1)
    # print(num_steps)

    # If steps has not been provided assume sampling has been done uniformly.
    if steps is None:
        steps = torch.arange(0, num_steps).unsqueeze(0).repeat([batch_size, 1])

    # print(steps.size())

    # If seq_lens has not been provided assume is equal to the size of the
    # time axis in the emebeddings.
    if seq_lens is None:
        seq_lens = torch.tensor(num_steps).unsqueeze(0).repeat([batch_size]).int()

    # print(seq_lens)

    # check if batch_size if consistent with emb etc
    assert batch_size == embs.size(0)
    assert num_steps == steps.size(1)
    assert batch_size == steps.size(0)

    if normalize_embeddings:
        embs = F.normalize(embs, dim=-1, p=2)

    if stochastic_matching:
        loss = compute_stochastic_alignment_loss(
            embs=embs,
            steps=steps,
            seq_lens=seq_lens,
            num_steps=num_steps,
            batch_size=batch_size,
            loss_type=loss_type,
            similarity_type=similarity_type,
            num_cycles=num_cycles,
            cycle_length=cycle_length,
            temperature=temperature,
            label_smoothing=label_smoothing,
            variance_lambda=variance_lambda,
            huber_delta=huber_delta,
            normalize_indices=normalize_indices)
    else:
        raise NotImplementedError
        # loss = compute_deterministic_alignment_loss(
        #     embs=embs,
        #     steps=steps,
        #     seq_lens=seq_lens,
        #     num_steps=num_steps,
        #     batch_size=batch_size,
        #     loss_type=loss_type,
        #     similarity_type=similarity_type,
        #     temperature=temperature,
        #     label_smoothing=label_smoothing,
        #     variance_lambda=variance_lambda,
        #     huber_delta=huber_delta,
        #     normalize_indices=normalize_indices)

    return loss

