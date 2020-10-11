# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stochastic alignment between sampled cycles in the sequences in a batch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F

from .losses import classification_loss
from .losses import regression_loss

def _align_single_cycle(cycle, embs, cycle_length, num_steps,
                        similarity_type, temperature):
    # choose from random frame
    n_idx = (torch.rand(1)*num_steps).long()[0]
    # n_idx = torch.tensor(8).long()

    # Create labels
    onehot_labels = torch.eye(num_steps)[n_idx]

    # Choose query feats for first frame.
    query_feats = embs[cycle[0], n_idx:n_idx + 1]
    num_channels = query_feats.size(-1)
    for c in range(1, cycle_length + 1):
        candidate_feats = embs[cycle[c]]
        if similarity_type == 'l2':
            mean_squared_distance = torch.sum((query_feats.repeat([num_steps, 1]) -
                                               candidate_feats) ** 2, dim=1)
            similarity = -mean_squared_distance
        elif similarity_type == 'cosine':
            similarity = torch.squeeze(torch.matmul(candidate_feats, query_feats.transpose(0, 1)))
        else:
            raise ValueError('similarity_type can either be l2 or cosine.')

        similarity /= float(num_channels)
        similarity /= temperature

        beta = F.softmax(similarity, dim=0).unsqueeze(1).repeat([1, num_channels])
        query_feats = torch.sum(beta * candidate_feats, dim=0, keepdim=True)

    return similarity.unsqueeze(0), onehot_labels.unsqueeze(0)

def _align(cycles, embs, num_steps, num_cycles, cycle_length,
           similarity_type, temperature):
  """Align by finding cycles in embs."""
  logits_list = []
  labels_list = []
  for i in range(num_cycles):
    logits, labels = _align_single_cycle(cycles[i],
                                         embs,
                                         cycle_length,
                                         num_steps,
                                         similarity_type,
                                         temperature)
    logits_list.append(logits)
    labels_list.append(labels)

  logits = torch.cat(logits_list, dim=0)
  labels = torch.cat(labels_list, dim=0)

  return logits, labels

def gen_cycles(num_cycles, batch_size, cycle_length=2):
    """Generates cycles for alignment.
    Generates a batch of indices to cycle over. For example setting num_cycles=2,
    batch_size=5, cycle_length=3 might return something like this:
    cycles = [[0, 3, 4, 0], [1, 2, 0, 3]]. This means we have 2 cycles for which
    the loss will be calculated. The first cycle starts at sequence 0 of the
    batch, then we find a matching step in sequence 3 of that batch, then we
    find matching step in sequence 4 and finally come back to sequence 0,
    completing a cycle.
    Args:
    num_cycles: Integer, Number of cycles that will be matched in one pass.
    batch_size: Integer, Number of sequences in one batch.
    cycle_length: Integer, Length of the cycles. If we are matching between
      2 sequences (cycle_length=2), we get cycles that look like [0,1,0].
      This means that we go from sequence 0 to sequence 1 then back to sequence
      0. A cycle length of 3 might look like [0, 1, 2, 0].
    Returns:
    cycles: Tensor, Batch indices denoting cycles that will be used for
      calculating the alignment loss.
    """
    sorted_idxes = torch.arange(batch_size).unsqueeze(0).repeat([num_cycles, 1])
    sorted_idxes = sorted_idxes.view([batch_size, num_cycles])
    cycles = sorted_idxes[torch.randperm(len(sorted_idxes))].view([num_cycles, batch_size])
    cycles = cycles[:, :cycle_length]
    cycles = torch.cat([cycles, cycles[:, 0:1]], dim=1)

    return cycles


def compute_stochastic_alignment_loss(embs,
                                      steps,
                                      seq_lens,
                                      num_steps,
                                      batch_size,
                                      loss_type,
                                      similarity_type,
                                      num_cycles,
                                      cycle_length,
                                      temperature,
                                      label_smoothing,
                                      variance_lambda,
                                      huber_delta,
                                      normalize_indices):

    cycles = gen_cycles(num_cycles, batch_size, cycle_length)
    logits, labels = _align(cycles, embs, num_steps, num_cycles, cycle_length,
                            similarity_type, temperature)

    if loss_type == 'classification':
        loss = classification_loss(logits, labels, label_smoothing)
    # elif 'regression' in loss_type:
    #     steps = tf.gather(steps, cycles[:, 0])
    #     seq_lens = tf.gather(seq_lens, cycles[:, 0])
    #     loss = regression_loss(logits, labels, num_steps, steps, seq_lens,
    #                            loss_type, normalize_indices, variance_lambda,
    #                            huber_delta)
    else:
        raise ValueError('Unidentified loss type %s. Currently supported loss '
                         'types are: regression_mse, regression_huber, '
                         'classification .'
                         % loss_type)
    return loss


