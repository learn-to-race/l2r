import torch.nn as nn
import numpy as np
import pdb

from baselines.il.il_utils import AttributeDict
from baselines.il.building_blocks import Conv, Branching, FC, Join

g_conf = AttributeDict()
g_conf.immutable(False)

g_conf.SENSORS = {'rgb': (3, 384, 512)}
g_conf.NUMBER_FRAMES_FUSION = 1
g_conf.INPUTS = np.arange(30)
g_conf.PRE_TRAINED = False
g_conf.TARGETS = ['steer', 'acceleration']

# From: https://github.com/felipecode/coiltraine/blob/29060ab5fd2ea5531686e72c621aaaca3b23f4fb/network/models/coil_icra.py
class CILModel(nn.Module):
    def __init__(self, model_params):
        super(CILModel, self).__init__()
        self._make_nn(model_params)

    def forward(self, x, a):
        """ ###### APPLY THE PERCEPTION MODULE """
        x, inter = self.perception(x)
        ## Not a variable, just to store intermediate layers for future vizualization
        #self.intermediate_layers = inter

        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(a)
        """ Join measurements and perception"""
        j = self.join(x, m)

        
        branch_outputs = self.branches(j)
        ## We only have one branch now.
        out = branch_outputs[0]
        #speed_branch_output = self.speed_branch(x)

        # We concatenate speed with the rest.
        return out #+ [speed_branch_output]

    def _make_nn(self, params):
        self.params = params

        number_first_layer_channels = 0

        for _, sizes in g_conf.SENSORS.items():
            number_first_layer_channels += sizes[0] * g_conf.NUMBER_FRAMES_FUSION

        # Get one item from the dict
        sensor_input_shape = next(iter(g_conf.SENSORS.values()))
        sensor_input_shape = [number_first_layer_channels, sensor_input_shape[1],
                              sensor_input_shape[2]]

        # For this case we check if the perception layer is of the type "conv"
        if 'conv' in params['perception']:
            perception_convs = Conv(params={'channels': [number_first_layer_channels] +
                                                          params['perception']['conv']['channels'],
                                            'kernels': params['perception']['conv']['kernels'],
                                            'strides': params['perception']['conv']['strides'],
                                            'dropouts': params['perception']['conv']['dropouts'],
                                            'end_layer': True})

            perception_fc = FC(params={'neurons': [perception_convs.get_conv_output(sensor_input_shape)]
                                                  + params['perception']['fc']['neurons'],
                                       'dropouts': params['perception']['fc']['dropouts'],
                                       'end_layer': False})

            self.perception = nn.Sequential(*[perception_convs, perception_fc])

            number_output_neurons = params['perception']['fc']['neurons'][-1]

        elif 'res' in params['perception']:  # pre defined residual networks
            resnet_module = importlib.import_module('network.models.building_blocks.resnet')
            resnet_module = getattr(resnet_module, params['perception']['res']['name'])
            self.perception = resnet_module(pretrained=g_conf.PRE_TRAINED,
                                             num_classes=params['perception']['res']['num_classes'])

            number_output_neurons = params['perception']['res']['num_classes']

        else:

            raise ValueError("invalid convolution layer type")

        self.measurements = FC(params={'neurons': [len(g_conf.INPUTS)] +
                                                   params['measurements']['fc']['neurons'],
                                       'dropouts': params['measurements']['fc']['dropouts'],
                                       'end_layer': False})

        self.join = Join(
            params={'after_process':
                         FC(params={'neurons':
                                        [params['measurements']['fc']['neurons'][-1] +
                                         number_output_neurons] +
                                        params['join']['fc']['neurons'],
                                     'dropouts': params['join']['fc']['dropouts'],
                                     'end_layer': False}),
                     'mode': 'cat'
                    }
         )

        self.speed_branch = FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                  params['speed_branch']['fc']['neurons'] + [1],
                                       'dropouts': params['speed_branch']['fc']['dropouts'] + [0.0],
                                       'end_layer': True})

        # Create the fc vector separatedely
        branch_fc_vector = []
        #pdb.set_trace()
        for i in range(params['branches']['number_of_branches']):
            branch_fc_vector.append(FC(params={'neurons': [params['join']['fc']['neurons'][-1]] +
                                                         params['branches']['fc']['neurons'] +
                                                         [len(g_conf.TARGETS)],
                                               'dropouts': params['branches']['fc']['dropouts'] + [0.0],
                                               'end_layer': True}))

        self.branches = Branching(branch_fc_vector)  # Here we set branching automatically

        if 'conv' in params['perception']:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.1)

