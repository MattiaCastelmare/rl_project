from .encoder import FeatureEncoder

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
                        curl_latent_dim):
        self.encoder = FeatureEncoder(obs_shape=obs_shape, 
                                      act_shape=action_shape,
                                      q_dim=encoder_feature_dim,
                                      act_hidden_dims=(encoder_feature_dim,),
                                      fdm_hidden_dims=(encoder_feature_dim,))
                                      
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