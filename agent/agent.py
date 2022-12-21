from .encoder import FeatureEncoder
from .sac import Actor, Critic
import torch.nn as nn
import torch
import numpy as np
from utils import center_crop_image, infoNCE, soft_update_params

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

        self.idm_lr = idm_lr
        self.fdm_lr = fdm_lr
        self.idm_update_freq = idm_update_freq
        self.fdm_update_freq = fdm_update_freq

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
                            a_dim=action_shape[0],
                            log_std_max=actor_log_std_max,
                            log_std_min=actor_log_std_min,
                            hidden_dim=hidden_dim
                          ).to(device)
        self.actor.apply(weight_init)

        self.critic = Critic( s_dim=encoder_feature_dim,
                              a_dim=action_shape[0],
                              hidden_dim=hidden_dim
                          ).to(device)
        self.critic.apply(weight_init)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.training = True

        # optimizers
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999))

        # to normalize the intrinsic reward
        self.max_extrinsic_reward = 1

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.encoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()


    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.from_numpy(obs).to(self.device) / 255.0
            obs = obs.unsqueeze(0)
            q = self.encoder.encode(obs)
            mu, _, _, _ = self.actor(q)
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        print("sample")
        if obs.shape[-1] != self.image_size:
            obs = center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = obs.unsqueeze(0)
            q = self.encoder.encode(obs)
            _, pi, _, _ = self.actor(q)
            print("done")
            return pi.cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, L, step, env_step):
        # intrinsic_weight
        intrinsic_decay = 2e-5
        # intrinsic weight
        C = 0.2 
        obs, action, reward, next_obs, done= replay_buffer.sample()
        
        # update max extrinsic reward
        max_reward = np.max(reward.cpu().numpy())
        if max_reward > self.max_extrinsic_reward:
            self.max_extrinsic_reward = max_reward

        if self.training:
            ri = self.encoder.compute_intrinsic_reward(obs, next_obs, action, self.max_extrinsic_reward)
            
            if step % self.log_interval == 0:
                L.log('train/batch_reward', reward.mean(), env_step)

            reward = reward + C*np.exp(-intrinsic_decay*env_step)*ri
        
        if step % self.log_interval == 0:
            L.log('train/total_reward', reward.mean(), env_step)

        self.update_critic(obs, action, reward, next_obs, done, L, step, env_step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step, env_step)

        if step % self.critic_target_update_freq == 0:
            soft_update_params(self.critic.Q1, self.critic.Q1_target, self.critic_tau)
            soft_update_params(self.critic.Q2, self.critic.Q2_target, self.critic_tau)
            soft_update_params(self.encoder.query_encoder, self.encoder.key_encoder,self.encoder_tau)
        
        if step % self.cpc_update_freq == 0:
            self.update_encoder(obs, next_obs, action, L, env_step)

    def update_Q(self, state, action, reward, new_state, done, L, step, env_step):
        with torch.no_grad():
            _, new_action, log_pi, _ = self.actor(new_state)
            Q1_target, Q2_target = self.critic(new_state, new_action, target=True)
            value_target = torch.min(Q1_target,
                                 Q2_target) - self.alpha.detach() * log_pi
            Q_target = reward + ((1-done) * self.discount * value_target)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action, target=False)
        Q1_loss = nn.functional.mse_loss(current_Q1, Q_target)
        Q2_loss = nn.functional.mse_loss(current_Q2, Q_target)
        
        if step % self.log_interval == 0:
            L.log('train_critic/loss', Q1_loss+Q2_loss, env_step)

        self.critic_optimizer.zero_grad()
        (Q1_loss-Q2_loss).backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, s, L, step, env_step):
        # detach encoder, so we don't update it with the actor loss
        _, action, log_pi, _= self.actor(s)
        actor_Q1, actor_Q2 = self.critic(s, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, env_step)
        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, env_step)
            L.log('train_alpha/value', self.alpha, env_step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_encoder(self, state, next_state, action, L, step):
        
        # TODO add random crop + CURL loss

        q = self.encoder.encode(state)
        ae = self.encoder.action_embedding(action)
        next_q_predicted = self.encoder.forward_dynamics(torch.cat((q, ae), axis=1))

        next_q = self.CURL.encode(next_state, target=True)

        fdm_loss = infoNCE(next_q_predicted,next_q,self.encoder.sim_metrics["dot"], self.device)

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        fdm_loss.backward()

        self.encoder_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/curl_loss', fdm_loss, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.encoder.state_dict(), '%s/encoder_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.encoder.load_state_dict(
            torch.load('%s/encoder_%s.pt' % (model_dir, step))
        )

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