from .encoder import FeatureEncoder
from .sac import Actor, Critic
import torch.nn as nn
import torch
import numpy as np

class Agent():

    def __init__(self,  obs_shape,
                        action_shape,
                        device,
                        hidden_dim,
                        discount,
                        init_temperature,
                        alpha_lr,
                        alpha_beta,
                        actor_lr,
                        actor_beta,
                        actor_log_std_min,
                        actor_log_std_max,
                        actor_update_freq,
                        critic_lr,
                        critic_beta,
                        critic_tau,
                        critic_target_update_freq,
                        encoder_type,
                        encoder_feature_dim,
                        encoder_action_dim,
                        encoder_lr,
                        idm_lr,
                        fdm_lr,
                        encoder_tau,
                        num_layers,
                        num_filters,
                        cpc_update_freq,
                        idm_update_freq,
                        fdm_update_freq,
                        log_interval,
                        detach_encoder,
                        # curl_latent_dim # not used
                        ):

        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.detach_encoder = detach_encoder

        self.encoder = FeatureEncoder(obs_shape=obs_shape, 
                                act_shape=action_shape,
                                q_dim=encoder_feature_dim,
                                a_dim=encoder_action_dim,
                                act_hidden_dims=(encoder_feature_dim,),
                                fdm_hidden_dims=(encoder_feature_dim,),
                                num_filters=num_filters,
                                num_layers=num_layers
                                ).to(device)
        self.encoder.apply(weight_init)

        self.actor = Actor( s_dim=encoder_feature_dim,
                            a_dim=encoder_action_dim,
                            log_std_max=actor_log_std_max,
                            log_std_min=actor_log_std_min,
                            hidden_dim=hidden_dim
        )
        self.actor.apply(weight_init)

        self.critic = Critic( s_dim=encoder_feature_dim,
                              a_dim=encoder_action_dim,
                              hidden_dim=hidden_dim
        )
        self.critic.apply(weight_init)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)



        # optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999))

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.encoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()


    def select_action(self, obs):
        raise NotImplementedError()

    def sample_action(self, obs):
        raise NotImplementedError()
    
    def update(self, replay_buffer, L, step, env_step):
        raise NotImplementedError()

    def save(self, model_dir, step):
        raise NotImplementedError()
    def load(self, model_dir, step):
        raise NotImplementedError()

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)