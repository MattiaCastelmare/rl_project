import torch.nn as nn
import torch
from utils import make_MLP,copy_params


class PixelEncoder(nn.Module):
    def __init__(self, obs_shape: tuple, 
                       dim_out: int, 
                       num_layers=2,
                       ksize=3, 
                       num_filters=32,
                       mlp_hidden_dims=(),
                       hidden_act=nn.ReLU,
                       out_act=None):
        super().__init__()

        self.obs_shape = obs_shape
        self.dim_out = dim_out
        self.num_layers = num_layers
        
        layers = []
        layers.append(nn.Conv2d(obs_shape[0], num_filters, ksize, stride=2))
        layers.append(hidden_act)
        
        for _ in range(num_layers - 1):
            layers.append(nn.Conv2d(num_filters, num_filters, ksize, stride=1))
            layers.append(nn.ReLU)
        self.cnn = nn.Sequential(layers)

        # get flattened size through a forward
        with torch.no_grad():
            _test = torch.zeros(self.dim_in).to(self.device)
            out = self.cnn(_test)
            self.v_shape = out.flatten().shape[0]

        self.mlp = make_MLP(self.v_shape, self.dim_out, mlp_hidden_dims, out_act).to(self.device)
        self.ln = nn.LayerNorm(self.dim_out) # helps with contrastive loss

    def forward(self, obs: torch.FloatTensor):
        cnn = self.cnn(obs)
        mlp = self.mlp(cnn)

        q = self.ln(mlp)

        return q  

class FeatureEncoder(nn.Module):
    def __init__(self,
                 obs_shape: tuple,
                 act_shape: tuple,
                 q_dim: int,
                 a_dim: int,
                 num_layers: int,
                 num_filters: int,
                 act_hidden_dims=(50,),
                 fdm_hidden_dims=(50,),
                 ):
        super().__init__()

        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.q_dim = q_dim
        self.a_dim = a_dim
        
        self.query_encoder = PixelEncoder(obs_shape=obs_shape, dim_out=q_dim, num_filters=num_filters, num_layers=num_layers)
        self.key_encoder = PixelEncoder(obs_shape=obs_shape, dim_out=q_dim, num_filters=num_filters, num_layers=num_layers)
        copy_params(copy_from=self.query_encoder, copy_to=self.key_encoder)
        for param in self.key_encoder.parameters(): 
            param.requires_grad = False # disable gradient computation for target network

        self.action_embedding = make_MLP(self.act_shape,self.a_dim,act_hidden_dims)
        self.forward_dynamics = make_MLP(self.q_dim+self.a_dim,self.q_dim,fdm_hidden_dims)

        self.W = nn.Parameter(torch.rand((q_dim,q_dim))) # for bilinear product
        self.sim_metrics = { # similarity metrics for contrastive loss
            # do the dot product on every pair: (k' k)
            "dot": lambda x,y: torch.einsum("ij, kj -> ik", x,y), 

            # do the bilinear product on every pair: (k' W  k)
            "bilinear": lambda x,y: torch.einsum("ij, kj, jj -> ik", x,y,self.W),
        }
        self.max_intrinsic = 1e-8 # the maximum intrinsic reward (for normalization)

    def encode(self, obs: torch.FloatTensor, target=False, grad=True):    
        if target:
            return self.key_encoder(obs)
        elif grad:
            return self.query_encoder(obs)
        else:
            with torch.no_grad(): return self.query_encoder(obs)
    
    def compute_intrinsic_reward(self, obs, next_obs, action, max_reward):
        with torch.no_grad(): # don't backprop through this
            
            q = self.encode(obs, target=True)
            ae = self.action_embedding(action)
            
            next_q_predict = self.forward_dynamics(torch.cat((q,ae),dim=1)) # predict new state
            next_q = self.encode(next_obs, target=True)

            # we try to use MSE as a dissimilarity metric
            prediction_error = (next_q_predict-next_q).pow(2).sum(dim=1).sqrt()

            max_error = max(prediction_error)
            if max_error > self.max_intrinsic:
                self.max_intrinsic = max_error

            ri = prediction_error*(max_reward/self.max_intrinsic_error)
            return ri.reshape(-1,1)