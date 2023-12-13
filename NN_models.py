import numpy.random as npr
import flax.linen as nn

#Batching data 
class DataStream():
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


#AE Classes
class Sigmoid_Dropout_Encoder(nn.Module):
    d_hidden: list
    latents: int
    dropout_rates: list
        
    @nn.compact
    def __call__(self, x):
        for i in range(len(self.d_hidden)):
            x = nn.sigmoid(nn.Dense(self.d_hidden[i])(x))
            x = nn.Dropout(rate=self.dropout_rates[i])(x, deterministic=True)
        x = nn.Dense(self.latents, name='f5')(x)
        return x

class Jibs_Encoder(nn.Module):
    d_hidden: list
    n_latents: int
    dropout_rates: list
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dropout(rate=self.dropout_rates[0])(nn.relu(nn.Dense(self.d_hidden[0])(x)), deterministic=True)
        x = nn.Dropout(rate=self.dropout_rates[1])(nn.sigmoid(nn.Dense(self.d_hidden[1])(x)), deterministic=True)
        x = nn.Dropout(rate=self.dropout_rates[2])(nn.relu(nn.Dense(self.d_hidden[2])(x)), deterministic=True)
        x = nn.softmax(nn.Dense(self.n_latents)(x))
        return x

class Sigmoid_Dropout_Decoder(nn.Module):
    d_hidden: list
    out_dim: int
    dropout_rates: list

    @nn.compact
    def __call__(self, z):
        for i in range(len(self.d_hidden))[::-1]:
            z = nn.sigmoid(nn.Dense(self.d_hidden[i])(z))
            z = nn.Dropout(rate=self.dropout_rates[i])(z, deterministic=True)
        z = nn.Dense(self.out_dim, name='f5')(z)
        return z

class Jibs_Decoder(nn.Module):
    d_hidden: list
    out_dim: int
    dropout_rates: list

    @nn.compact
    def __call__(self, z):
        z = nn.Dropout(rate=self.dropout_rates[2])(nn.relu(nn.Dense(self.d_hidden[2])(z)), deterministic=True)
        z = nn.Dropout(rate=self.dropout_rates[1])(nn.sigmoid(nn.Dense(self.d_hidden[1])(z)), deterministic=True)
        z = nn.Dropout(rate=self.dropout_rates[0])(nn.relu(nn.Dense(self.d_hidden[0])(z)), deterministic=True)
        z = nn.Dense(self.out_dim, name='f5')(z)
        return z

class Sigmoid_Dropout_AutoEncoder(nn.Module):
    input_size: int
    hidden_layers: tuple
    n_latents: int
    dropout_rates: list

    def setup(self):
        self.encoder = Sigmoid_Dropout_Encoder(list(self.hidden_layers), self.n_latents, self.dropout_rates)
        self.decoder = Sigmoid_Dropout_Decoder(list(self.hidden_layers), self.input_size, self.dropout_rates)

    def __call__(self, x, z_rng):
        z_latent = self.encoder(x)
        return self.decoder(z_latent), z_latent

    def decode(self, z, rng):
        return self.decoder(z)

class Jibs_AutoEncoder(nn.Module):
    input_size: int
    hidden_layers: tuple
    n_latents: int
    dropout_rates: list

    def setup(self):
        self.encoder = Jibs_Encoder(list(self.hidden_layers), self.n_latents, self.dropout_rates)
        self.decoder = Jibs_Decoder(list(self.hidden_layers), self.input_size, self.dropout_rates)

    def __call__(self, x, z_rng):
        z_latent = self.encoder(x)
        return self.decoder(z_latent), z_latent

    def decode(self, z, z_rng):
        return self.decoder(z)
