import numpy.random as npr
import flax
import flax.linen as nn

##############################################################################
# Last Amended February 12, 2024 
#      Author J.A.DePaolo-Boisvert
##############################################################################
# A module for variable creation of AutoEncoders
# Currently limited to creating symmetric encoding-decoding pairs
# Parameters for these should be set in the json input file, as
# shown below.  hidden layers is typically chosen to be the same
# as the input size, making all Dense Layers except those immediately
# adjacent to the latents square matrix transformations
#     "training" : {
#        "arch" : {
#            "latent_dim" : 7,
#            "dropout_rates" : [0.2, 0.2, 0.2],
#            "hidden_layers" : [306, 306, 306],
#            "net_layers" : ["flax.linen.Dense", "flax.linen.Dense", "flax.linen.Dense", "flax.linen.Dense"],
#            "activators" : ["flax.linen.relu", "flax.linen.relu", "flax.linen.relu", "flax.linen.relu"]},
#
#  The above example would create an encoding decoding pair with
#    following passthrough
#    Encoder: Input -> (Input_size)Dense(306) -> Relu -> (306)Dense(306) ->
#             Relu -> (306)Dense(306) -> Relu -> (306)Dense(7) -> ReLu -> Latent Data
#    Decoder: Latent Data -> (7)Dense(306) -> Relu -> (306)Dense(306) -> Relu ->
#             (306)Dense(306) -> Relu -> (306)Dense(Input_Size) -> ReLu -> Output
#
##############################################################################


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

class Encoder(nn.Module):
    '''
    Encoder module for a neural network.
    Args:
        d_hidden (list): List of hidden layer sizes.
        latent_size (int): Size of latent space.
        activators (list): List of activation functions for each layer.
        layer_ops (list): List of layer operations for each layer.
        dropout_rates (list): List of dropout rates for each layer.
    Returns:
        Encoded representation of the input.

    '''
    d_hidden: list
    latent_size: int
    activators: list
    layer_ops: list
    dropout_rates: list
        
    @nn.compact
    def __call__(self, x): 
        '''
        Encodes the input data.
        Args:
            x: Input data.
        Returns:
            Latent representation of the input data.

        '''
        for i in range(len(self.d_hidden)):
            x = self.activators[i](self.layer_ops[i](self.d_hidden[i])(x))
            x = nn.Dropout(rate=self.dropout_rates[i])(x, deterministic=True)
        x = self.activators[-1](self.layer_ops[-1](self.latent_size)(x))
        return x

class Decoder(nn.Module):
    '''
    Decoder module for a neural network.
    Args:
        d_hidden (list): List of hidden layer sizes.
        latent_size (int): Size of latent space.
        activators (list): List of activation functions for each layer.
        layer_ops (list): List of layer operations for each layer.
        dropout_rates (list): List of dropout rates for each layer.
    Returns:
        Decoded representation of the output.

    '''
    d_hidden: list
    output_size: int
    activators: list
    layer_ops: list
    dropout_rates: list

    @nn.compact
    def __call__(self, z):
        '''
        Decodes the latent representation.
        Args:
            z: Latent representation.
        Returns:
            Decoded latent representation into original space.

        '''
        for i in range(len(self.d_hidden))[::-1]:
            z = self.activators[i](self.layer_ops[i](self.d_hidden[i])(z))
            z = nn.Dropout(rate=self.dropout_rates[i])(z, deterministic=True)
        z = self.activators[0](self.layer_ops[0](self.output_size)(z))
        return z

class AutoEncoder(nn.Module):
    '''
    Autoencoder neural network model, composed of both an encoder and decoder.
    Args:
        input_size (int): Size of the input data.
        n_latents (int): Size of the latent space.
        hidden_layers (list): List of hidden layer sizes.
        activators (list): List of activation functions for each layer.
        layer_ops (list): List of layer operations for each layer.
        dropout_rates (list): List of dropout rates for each layer.
    Methods:
        setup(): Initializes the encoder and decoder modules.
        __call__(x, z_rng): Encodes the input data and then decodes it back to the original space.
        decode(z, z_rng): Decodes the latent representation back to the original space.
    Example:
        autoencoder = AutoEncoder(**args**)
        output, latent_representation = autoencoder(input_data, z_rng)

    '''
    input_size: int
    n_latents: int
    hidden_layers: list
    activators: list
    layer_ops: list
    dropout_rates: list

    def setup(self):
        '''
        Initializes the encoder and decoder modules.

        '''
        self.encoder = Encoder(self.hidden_layers, self.n_latents, self.activators, self.layer_ops, self.dropout_rates)
        self.decoder = Decoder(self.hidden_layers, self.input_size, self.activators, self.layer_ops, self.dropout_rates)

    def __call__(self, x, z_rng):
        '''
        Encodes the input data and then decodes back to original space.

        '''
        z_latent = self.encoder(x)
        return self.decoder(z_latent), z_latent

    def decode(self, z, z_rng):
        '''
        Decodes the latent representation back to the original space.
        Args:
            z: Latent representation.
            z_rng: Random number generator.
        Returns:
            Decoded output of the latent representation.

        '''
        return self.decoder(z)
