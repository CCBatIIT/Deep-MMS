import numpy.random as npr
import flax, jax
import flax.linen as nn
import jax.numpy as jnp

##############################################################################
# Last Amended September 11, 2024
#      Author J.A.DePaolo-Boisvert
##############################################################################
# A module for variable creation of AutoEncoders
# Currently limited to creating symmetric encoding-decoding pairs
# Parameters for these should be set in the json input file, as
# shown below.  hidden layers is typically chosen to be the same
# as the input size, making all Dense Layers except those immediately
# adjacent to the latents square matrix transformations
##############################################################################

#Batching data 
class Data_stream():
    def __init__(self, rng_seed, num_total, num_batches, batch_size, data):
        self.rng_seed = rng_seed
        self.num_total = num_total
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.data = data
        
    def __iter__(self):
        rng = npr.RandomState(self.rng_seed)
        while True:
            perm = rng.permutation(self.num_total)
            for i in range(self.num_batches):
                batch_idx = perm[i * self.batch_size:(i + 1) * self.batch_size]
                yield self.data[batch_idx]

#Main Use Classes
class Encoder(nn.Module):
    d_hidden: list
    latents: int
    dropout_rates: list

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.d_hidden)):
            x = nn.relu(nn.Dense(self.d_hidden[i])(x))
            x = nn.Dropout(rate=self.dropout_rates[i])(x, deterministic=True)
        x = nn.Dense(self.latents, name='f5')(x)
        return x
    
class Decoder(nn.Module):
    d_hidden: list
    out_dim: int
    dropout_rates: list

    @nn.compact
    def __call__(self, z):
        for i in range(len(self.d_hidden))[::-1]:
            z = nn.relu(nn.Dense(self.d_hidden[i])(z))
            z = nn.Dropout(rate=self.dropout_rates[i])(z, deterministic=True)
        z = nn.Dense(self.out_dim, name='f5')(z)
        return z
        
class AE(nn.Module):
    input_size: int
    hidden_layers: tuple
    dropout_rates: list
    latents: int

    def setup(self):
        self.encoder = Encoder(list(self.hidden_layers), self.latents, self.dropout_rates)
        self.decoder = Decoder(list(self.hidden_layers), self.input_size, self.dropout_rates)
    
    def __call__(self, x, z_rng):
        z_latent = self.encoder(x)
        return self.decoder(z_latent), z_latent
    
    def encode(self, x, z_rng):
        return self.encoder(x)
    
    def decode(self, z, z_rng):
        return self.decoder(z)
        
#Extra (Not in Use Classes)

def reparameterize(z_rng, z_mean, z_logvar):
    z_std = jnp.exp(0.5*z_logvar)
    z_eps = jax.random.normal(z_rng, z_logvar.shape)
    return z_mean + z_eps*z_std


class BatchNorm_VAE(nn.Module):
    input_size: int
    hidden_layers: tuple
    latents: int
        
    def setup(self):
        self.encoder = BVEncoder(list(self.hidden_layers), self.latents)
        self.decoder = BVDecoder(list(self.hidden_layers), self.input_size)

    def __call__(self, x, z_rng, train:bool):
        z_mean, z_logvar = self.encoder(x, train=train)
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decoder(z, train=train), z_mean, z_logvar
    
    def construct(self, z_mean, z_logvar, z_rng, train=False):
        z = reparameterize(z_rng, z_mean, z_logvar)
        return self.decoder(z, train=train)
    
    def encode(self, x, z_rng, train=False):
        return self.encoder(x, train=train)
    
    def decode(self, z, z_rng, train=False):
        return self.decoder(z, train=train)

class BVEncoder(nn.Module):
    d_hidden: list
    latents: int
    
    @nn.compact
    def __call__(self, x, train: bool):
        for i in range(len(self.d_hidden)):
            x = nn.relu(nn.Dense(self.d_hidden[i])(x))
            x = nn.BatchNorm(use_running_average=not train)(x)
        mean_x = nn.Dense(self.latents, name='fc5_mean')(x)
        logvar_x = nn.Dense(self.latents, name='fc5_logvar')(x)
        return mean_x, logvar_x 

class BVDecoder(nn.Module):
    d_hidden: list
    out_dim: int

    @nn.compact
    def __call__(self, z, train: bool):
        for i in range(len(self.d_hidden))[::-1]:
            z = nn.relu(nn.Dense(self.d_hidden[i])(z))
            z = nn.BatchNorm(use_running_average=not train)(z)
        z = nn.Dense(self.out_dim, name='f5')(z)
        return z

