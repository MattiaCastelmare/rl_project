import torch.nn as nn
import torch
from utils import make_MLP, copy_params
import numpy as np

def gaussian_logprob(self,noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)

def squash(mu, pi, log_pi):
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(nn.functional.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi

class Actor(nn.Module):
    def __init__(self, 
                 s_dim: int,
                 a_dim: int,
                 hidden_dim: int,
                 log_std_min: float,
                 log_std_max: float
                 ):
        super().__init__()

        self.policy_network = make_MLP(s_dim,2*a_dim,(hidden_dim,hidden_dim,))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    
    def forward(self, s, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        # split output in two
        mu, log_std = self.policy_network(s).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        # reparametrization trick
        noise = torch.randn_like(mu)
        pi = mu + noise * log_std.exp() 

        log_pi = gaussian_logprob(noise, log_std)

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, s_dim, a_dim, hidden_dim
        ):
        super().__init__()

        self.Q1 = make_MLP(s_dim+a_dim, (hidden_dim,hidden_dim))
        self.Q2 = make_MLP(s_dim+a_dim, (hidden_dim,hidden_dim))

        self.Q1_target = make_MLP(s_dim+a_dim, (hidden_dim,hidden_dim))
        self.Q2_target = make_MLP(s_dim+a_dim, (hidden_dim,hidden_dim))
        copy_params(copy_from=self.Q1, copy_to=self.Q1_target)
        for param in self.Q1_target.parameters(): 
            param.requires_grad = False # disable gradient computation for target network
        copy_params(copy_from=self.Q2, copy_to=self.Q2_target)
        for param in self.Q2_target.parameters(): 
            param.requires_grad = False # disable gradient computation for target network


    def forward(self, s, a, target=False):
        if target:
            q1 = self.Q1_target(torch.cat((s,a),dim=1))
            q2 = self.Q2_target(torch.cat((s,a),dim=1))
        else:
            q1 = self.Q1(torch.cat((s,a),dim=1))
            q2 = self.Q2(torch.cat((s,a),dim=1))
        return q1, q2