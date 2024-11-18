import torch.nn as nn
import numpy as np
import torch
import time
import torch.optim as optim
from rsl_rl.modules import ActorCritic
from torch.distributions import Normal

class TCN_encoder(nn.Module):
    def __init__(self, activation,
                conv_dims = [(42, 32, 6, 5), (32, 16, 4, 2)],
                # input_channel, output_channel, kernal_size, stride
                period_length = 100,):
        super(TCN_encoder, self).__init__()
        self.output_dim = period_length
        self.activation_fn = self.get_activation(activation)
        self.output_size = period_length

        encoder_layers = []
        for input_chl, output_chl, kernal_size, stride in conv_dims:
            c1 = nn.Conv1d(input_chl, output_chl, kernal_size, stride=stride)
            torch.nn.init.orthogonal_(c1.weight, gain=np.sqrt(2))
            encoder_layers.append(c1)
            encoder_layers.append(self.activation_fn)
            self.output_size = (self.output_size-kernal_size) // stride + 1
        encoder_layers.append(nn.Flatten())


        self.encoder = nn.Sequential(*encoder_layers)
        self.input_channel, _, _, _ = conv_dims[0]
        _, self.output_channel, _, _ = conv_dims[-1]
        self.period_length = period_length
        self.output_size *= self.output_channel 
        print("Encoder output size: ", self.output_size)

        # self.latent_mu = nn.Linear(self.output_size, self.output_size//4)
        # self.latent_var = nn.Linear(self.output_size, self.output_size//4)

        # self.h1 = nn.Conv1d(30, 32, 6, stride=5)
        # torch.nn.init.orthogonal_(self.h1.weight, gain=np.sqrt(2))

        # self.h2 = nn.Conv1d(32, 16, 4, stride=2)
        # torch.nn.init.orthogonal_(self.h2.weight, gain=np.sqrt(2))



        
    def get_activation(self, act_name):
        if act_name == "elu":
            return nn.ELU()
        elif act_name == "selu":
            return nn.SELU()
        elif act_name == "relu":
            return nn.ReLU()
        elif act_name == "crelu":
            return nn.ReLU()
        elif act_name == "lrelu":
            return nn.LeakyReLU()
        elif act_name == "tanh":
            return nn.Tanh()
        elif act_name == "sigmoid":
            return nn.Sigmoid()
        else:
            print("invalid activation function!")
            return None
    
    def forward(self, x):
        # x = obs_long.view(-1,30,100)
        # # x = x.transpose(1,2)
        # x = self.h1(x)
        # x = self.relu(x)
        # x = self.h2(x)
        # x = self.relu(x)
        # x = x.view(x.size(0), -1)
        # x = torch.cat((x, obs_short),dim=1)
        # x = self.actor(x)
        y = self.encoder(x)
        return y
    
    def encode_latent(self, x):
        y = self.encoder(x)
        mu = self.latent_mu(y)
        var = self.latent_var(y)
        return mu, var

class TCN_decoder(nn.Module):
    def __init__(self, activation, input_dim, output_dim, hidden_dims=[256, 256, 256]):
        super(TCN_decoder, self).__init__()
        self.activation_fn = self.get_activation(activation)
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(self.activation_fn)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(self.activation_fn)
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # x = obs_long.view(-1,30,100)
        # # x = x.transpose(1,2)
        # x = self.h1(x)
        # x = self.relu(x)
        # x = self.h2(x)
        # x = self.relu(x)
        # x = x.view(x.size(0), -1)
        # x = torch.cat((x, obs_short),dim=1)
        # x = self.actor(x)
        y = self.decoder(x)
        return y

    def get_activation(self, act_name):
        if act_name == "elu":
            return nn.ELU()
        elif act_name == "selu":
            return nn.SELU()
        elif act_name == "relu":
            return nn.ReLU()
        elif act_name == "crelu":
            return nn.ReLU()
        elif act_name == "lrelu":
            return nn.LeakyReLU()
        elif act_name == "tanh":
            return nn.Tanh()
        elif act_name == "sigmoid":
            return nn.Sigmoid()
        else:
            print("invalid activation function!")
            return None
        

class ActorCriticTCN(ActorCritic):
    def __init__(self, num_actor_obs,
                num_critic_obs,
                num_actions,
                activation='elu',
                actor_hidden_dims=[256, 256, 256],
                critic_hidden_dims=[256, 256, 256],
                init_noise_std=1.0,
                custom_actor_args={},
                custom_critic_args={},
                #TCN parameters
                TCN_activation = 'relu',
                conv_dims = [(48, 32, 6, 5), (32, 16, 4, 2)],
                period_length = 100,
                mean_sigma_coef = 0.0,
                **kwargs):

        encoder = TCN_encoder(TCN_activation, conv_dims, period_length)
        super().__init__(encoder.output_size + (num_actor_obs - encoder.input_channel * encoder.period_length),
                num_critic_obs,
                num_actions,
                activation=activation,
                actor_hidden_dims=actor_hidden_dims,
                critic_hidden_dims=critic_hidden_dims,
                init_noise_std=init_noise_std,
                custom_actor_args=custom_actor_args,
                custom_critic_args=custom_critic_args,
                **kwargs)
        self.encoder = encoder
        # self.decoder = TCN_decoder(TCN_activation, self.encoder.output_size//4, (3*num_actions+6), hidden_dims=[256, 256, 256])
        self.num_actor_obs = num_actor_obs
        self.mean_sigma_coef = mean_sigma_coef

        print("------------------------------------------")
        print("using TCN encoder")
        print("------------------------------------------")


    # def act(self, observations, **kwargs):
    #     obs_buf = observations[:, :(self.num_actor_obs - self.encoder.input_channel * self.encoder.period_length)]
    #     obs_history = observations[:, self.num_actor_obs - self.encoder.input_channel * self.encoder.period_length:].view(-1, self.encoder.input_channel, self.encoder.period_length)
    #     encoded_hist = self.encoder(obs_history)
    #     input = torch.cat((obs_buf, encoded_hist), dim=-1)
    #     self.update_distribution(input)
    #     return self.distribution.sample()
    
    def update_distribution(self, observations):
        obs_buf = observations[:, :(self.num_actor_obs - self.encoder.input_channel * self.encoder.period_length)]
        obs_history = observations[:, self.num_actor_obs - self.encoder.input_channel * self.encoder.period_length:].view(-1, self.encoder.period_length, self.encoder.input_channel).transpose(1,2)
        encoded_hist = self.encoder(obs_history)
        input = torch.cat((obs_buf, encoded_hist+0.04*torch.randn_like(encoded_hist)), dim=-1)
        mean = self.actor(input)
        # self.distribution = Normal(mean, mean*0. + self.std)
        self.distribution = Normal(mean, torch.abs(mean)*self.mean_sigma_coef + self.std)
    
    def act_inference(self, observations):
        obs_buf = observations[:, :(self.num_actor_obs - self.encoder.input_channel * self.encoder.period_length)]
        obs_history = observations[:, self.num_actor_obs - self.encoder.input_channel * self.encoder.period_length:].view(-1, self.encoder.period_length, self.encoder.input_channel).transpose(1,2)
        encoded_hist = self.encoder(obs_history)
        input = torch.cat((obs_buf, encoded_hist), dim=-1)
        actions_mean = self.actor(input)

        return actions_mean
    
    def predict_from_hist(self, observations):
        obs_history = observations[:, self.num_actor_obs - self.encoder.input_channel * self.encoder.period_length:].view(-1, self.encoder.period_length, self.encoder.input_channel).transpose(1,2)
        latent_mu, latent_var = self.encoder.encode_latent(obs_history)
        std = torch.exp(0.5 * latent_var)
        eps = torch.randn_like(std)
        encoded_latent = eps * std + latent_mu
        return self.decoder(encoded_latent)
    
    def kld_loss(self, observations):
        obs_history = observations[:, self.num_actor_obs - self.encoder.input_channel * self.encoder.period_length:].view(-1, self.encoder.period_length, self.encoder.input_channel).transpose(1,2)
        latent_mu, latent_var = self.encoder.encode_latent(obs_history)
        kld_loss = -0.5 * torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp())

        return kld_loss









# mlp_input_dim_a = 167
# activation = nn.ReLU()
# actor_hidden_dims = [256, 256, 256]
# num_actions = 12
# m = TCN_encoder(mlp_input_dim_a, activation, actor_hidden_dims, num_actions)

# obs_short = torch.randn(100, 167)
# obs_long = torch.randn(100, 30, 100)
# t=time.time()

# output = m(obs_short, obs_long)
# print(time.time()-t)
# print(output.shape)

# x = torch.rand((10,30,100))
# obs = torch.rand((10,165))
# obs = torch.rand((10,3165))
# obs = torch.arange(1, 71)
# obs = obs.view(2,7,5).transpose(1,2)
# print(obs[:,:,:])

# my_act = ActorCriticTCN(165,165,12)
# y = my_act.act(obs)
# print(y.size())

if __name__ == '__main__':
#     obs = torch.arange(1, 71)
#     obs = obs.view(2,7,5).transpose(1,2)
#     print(obs[:,:,:])
    loss = nn.MSELoss()
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.randn(3, 5)
    output = loss(input, target)
    print(output)
    print(output.mean(-1))
    output.backward()

