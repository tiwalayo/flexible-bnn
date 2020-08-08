from src.models.stochastic.mc_dropout.models import LinearNetwork as LinearNetworkMC
from src.models.stochastic.mc_dropout.models import ConvNetwork as ConvNetworkMC

from src.models.stochastic.bbb.models import LinearNetwork as LinearNetworkBBB
from src.models.stochastic.bbb.models import ConvNetwork as ConvNetworkBBB

from src.models.stochastic.sgld.models import LinearNetwork as LinearNetworkSGLD
from src.models.stochastic.sgld.models import ConvNetwork as ConvNetworkSGLD

STOCHASTIC_FACTORY = {"linear_mc": lambda input_size, output_size, layers, activation, args: LinearNetworkMC(input_size, output_size, layers, activation, args),
                     "conv_mc": lambda input_size, output_size, layers, activation, args: ConvNetworkMC(input_size, output_size, layers, activation, args),
                     "linear_bbb": lambda input_size, output_size, layers, activation, args: LinearNetworkBBB(input_size, output_size, layers, activation, args),
                     "conv_bbb": lambda input_size, output_size, layers, activation, args: ConvNetworkBBB(input_size, output_size, layers, activation, args),
                     "linear_sgld": lambda input_size, output_size, layers, activation, args, training_mode: LinearNetworkSGLD(input_size, output_size, layers, activation, args, training_mode),
                     "conv_sgld": lambda input_size, output_size, layers, activation, args, training_mode: ConvNetworkSGLD(input_size, output_size, layers, activation, args, training_mode)}
