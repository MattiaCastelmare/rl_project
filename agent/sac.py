import torch.nn as nn
import torch
from utils import make_MLP, copy_params, soft_update_params
from torch.distributions import Normal
import numpy as np
# soft actor critic for vector states
class SAC(nn.Module):
    def __init__(self,
                s_dim,
                a_dim,
                Q_hidden_dims: tuple,
                policy_hidden_dims: tuple,
                gamma: float,
                tau: float,
                log_std_bounds: tuple,
                alpha: float,
                actor_lr = int,
                Q1_lr = int,
                Q2_lr = int,
                epsilon = float,

                ):
        super().__init__()
        self.Q_network1 = make_MLP(s_dim + a_dim, 1, Q_hidden_dims)
        self.Q_network2 = make_MLP(s_dim + a_dim, 1, Q_hidden_dims)

        self.Q_target1 = make_MLP(s_dim + a_dim, 1, Q_hidden_dims)
        self.Q_target2 = make_MLP(s_dim + a_dim, 1, Q_hidden_dims)

        for param1, param2 in zip(self.Q_target1.parameters(),self.Q_target2.parameters()): 

            param1.requires_grad = False # disable gradient computation for target network
            param2.requires_grad = False
        # copy params at start ? Yes as written in OpenAI pseudocode https://spinningup.openai.com/en/latest/algorithms/sac.html (M) 
        copy_params(self.Q_network1, self.Q_target1)
        copy_params(self.Q_network2, self.Q_target2)

        self.policy_network = make_MLP(s_dim, 2* a_dim, policy_hidden_dims) 
                                              # half for the mean
                                              # and half for the (log) std

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.log_std_bounds = log_std_bounds
        self.a_dim = a_dim
        self.epsilon = epsilon

        self.critic1_loss = nn.MSELoss()
        self.critic2_loss = nn.MSELoss()

        self.actor_optimizer = torch.optim.Adam(self.policy_network.parameters(),
                                                lr=actor_lr)
        self.Q_network1_optimizer = torch.optim.Adam(self.Q_network1.parameters(),
                                                lr=Q1_lr)
        self.Q_network2_optimizer = torch.optim.Adam(self.Q_network2.parameters(),
                                                lr=Q2_lr)                                                                      
    
    def rep_trick(self, mu, std):
        normal = Normal(0, 1)
        z = normal.sample()
        return torch.tanh(mu + std*z)

    def check_tensor(self, x):
        if type(x) != torch.Tensor:
            x = np.array(x)
            x = torch.from_numpy(x)
        return x


    def policy_forward(self, state):
        state = self.check_tensor(state)
        
        out = self.policy_network(state)

        if len(out) == 2*self.a_dim:

            mu = out[:self.a_dim]
            
            log_std = out[self.a_dim:]
            
        else:
            mu = out[:, self.a_dim - 1]
            log_std = out[:,-1]

        
        
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds

        # bound the log_std between min and max
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
                                                                             
        std = log_std.exp()

        return mu, std

    def Qnet_forward(self, state, action):
        state = self.check_tensor(state)
        action = self.check_tensor(action)

        obs_action = torch.cat([state, action], dim = -1)
        Q1 = self.Q_network1(obs_action)
        Q2 = self.Q_network2(obs_action)
        return Q1, Q2

    def Qtarg_forward(self, state, action):
        state = self.check_tensor(state)
        action = self.check_tensor(action)
       
        obs_action = torch.cat([state, action], dim = -1 )

        Q1 = self.Q_target1(obs_action)
        Q2 = self.Q_target2(obs_action)
        return Q1, Q2

    def actor(self, state):
        mu, std = self.policy_forward(state)
        dist = self.rep_trick(mu, std)

        return dist

    def return_log(self, state):
        mu, std = self.policy_forward(state)

        normal = Normal(0, 1)
        z = normal.sample()
        action = self.rep_trick(mu, std)
        
        log_prob = Normal(mu, std).log_prob(mu+ std*z) - torch.log(1 - action.pow(2) + self.epsilon)
        
        return log_prob, action.unsqueeze(-1)

    def get_action(self,state):
        state = self.check_tensor(state)
        dist = self.actor(state)
        action  = dist.cpu().detach().numpy()
        return action


    def alpha_decay(self):
        self.alpha*=0.9

    def update_policy(self, state, Q_net1, Q_net2):

        log_prob, _ = self.return_log(state)
        actor_Q = torch.min(Q_net1, Q_net2)

        # minus because we perform a gradient descent
        actor_loss = -(actor_Q.detach() - self.alpha*log_prob).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def update_Qnetworks(self, reward, done, Q_targ1, Q_targ2, Q_net1, Q_net2, log_prob):
        Q_critic = torch.min(Q_targ1, Q_targ2)
        
        target = reward + self.gamma*(1-done)*(Q_critic - self.alpha*log_prob)

        critic1_loss = self.critic1_loss(Q_net1, target.detach())
        self.Q_network1_optimizer.zero_grad()
        critic1_loss.backward()
        self.Q_network1_optimizer.step()

        critic2_loss = self.critic1_loss(Q_net2, target.detach())
        self.Q_network2_optimizer.zero_grad()
        critic2_loss.backward()
        self.Q_network2_optimizer.step()

        return critic1_loss, critic2_loss
    def update_SAC(self, state, reward, action, new_state, done):
        state      = torch.FloatTensor(state)
        new_state  = torch.FloatTensor(new_state)
        action     = torch.FloatTensor(action)
        reward     = torch.FloatTensor(reward).unsqueeze(1)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1)
        log_prob, new_action = self.return_log(new_state)

        Q_net1, Q_net2 = self.Qnet_forward(state, action)
        
        Q_targ1, Q_targ2 = self.Qtarg_forward(new_state, new_action)
        
        loss1, loss2 = self.update_Qnetworks(reward, done, Q_targ1, Q_targ2, Q_net1, Q_net2, log_prob)

        dist = self.actor(state)
        dist = dist.unsqueeze(-1)
        Q1, Q2 = self.Qnet_forward(state, dist)
        loss3 = self.update_policy(state, Q1, Q2)

        soft_update_params(self.Q_network1, self.Q_target1,0.01)
        soft_update_params(self.Q_network2, self.Q_target2,0.01)       

        return loss1.item() + loss2.item() + loss3.item() 
