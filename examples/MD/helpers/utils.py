import sys
import torch
from torch.autograd import Variable
import numpy as np
from absl import app

# Metric function
def metric(displacement_fn):
    def metric_fn(Ra, Rb):
        dR = displacement_fn(Ra, Rb)
        return torch.sum(dR * dR, axis=-1)

    return metric_fn


# Soft sphere pair energy function
def soft_sphere_pair(displacement_fn, species, sigma):
    def energy_fn(R):
        dR = displacement_fn(R[:, None, :], R[None, :, :])
        dR_mag = torch.sqrt(torch.sum(dR * dR, axis=-1))

        sigma_ij = sigma[species][:, None] * sigma[species][None, :]
        energy = torch.where(dR_mag < sigma_ij, 0.5 * (1 - dR_mag / sigma_ij) ** 2, torch.zeros_like(dR_mag))

        return torch.sum(energy) / R.shape[0]

    return energy_fn


# Force function
def force(energy_fn):
    def force_fn(R):
        R = Variable(R, requires_grad=True)
        E = energy_fn(R)
        E.backward(torch.ones_like(E))
        return R.grad

    return force_fn
