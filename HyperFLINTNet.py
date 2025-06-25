import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
from model.refine import *
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from utils import plot_loss, visualize_ind, visualize_series, visualize_series_flow, visualize_large

# class SineActivation(nn.Module):
#     def forward(self, x):
#         return torch.sin(x)

# import torch.nn.init as init

# def sine_init(m):
#     if isinstance(m, nn.Linear):
#         num_input = m.weight.size(-1)
#         with torch.no_grad():
#             m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))
#     elif isinstance(m, nn.Conv1d):
#         num_input = m.weight.size(1)
#         with torch.no_grad():
#             m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))
#     elif isinstance(m, nn.Conv3d):
#         num_input = m.weight.size(1) * m.weight.size(2) * m.weight.size(3) * m.weight.size(4)
#         with torch.no_grad():
#             m.weight.uniform_(-np.sqrt(6 / num_input), np.sqrt(6 / num_input))

# def first_layer_sine_init(m):
#     if isinstance(m, nn.Linear):
#         num_input = m.weight.size(-1)
#         with torch.no_grad():
#             m.weight.uniform_(-1 / num_input, 1 / num_input)
#     elif isinstance(m, nn.Conv1d):
#         num_input = m.weight.size(1)
#         with torch.no_grad():
#             m.weight.uniform_(-1 / num_input, 1 / num_input)
#     elif isinstance(m, nn.Conv3d):
#         num_input = m.weight.size(1) * m.weight.size(2) * m.weight.size(3) * m.weight.size(4)
#         with torch.no_grad():
#             m.weight.uniform_(-1 / num_input, 1 / num_input)

# def kaiming_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
#     elif isinstance(m, nn.Conv1d):
#         nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
#     elif isinstance(m, nn.Conv3d):
#         nn.init.kaiming_normal_(m.weight, nonlinearity='linear')
#     elif isinstance(m, nn.ConvTranspose3d):
#         nn.init.kaiming_normal_(m.weight, nonlinearity='linear')

# def prelu_init(m):
#     if isinstance(m, nn.PReLU):
#         m.weight.data.fill_(0.25)


def functional_conv3d(x, weight, bias, out_planes, stride=1, padding=1, dilation=1):
    x = F.conv3d(x, weight, bias, stride=stride, padding=padding, dilation=dilation)
    prelu = nn.PReLU(num_parameters=out_planes).to(x.device)
    # prelu = nn.SELU().to(x.device)
    # prelu = SineActivation().to(x.device)
    x = prelu(x)
    return x

def functional_conv_transpose3d(x, weight, bias, out_planes, stride=2, padding=1):
    x = F.conv_transpose3d(x, weight, bias, stride=stride, padding=padding)
    prelu = nn.PReLU(num_parameters=out_planes).to(x.device)
    # prelu = nn.SELU().to(x.device)
    # prelu = SineActivation().to(x.device)
    x = prelu(x)
    return x

# """
# autoencoder: instead of pixels we have weights
# k channels times 3x3 spatial dims
class HyperNetwork(nn.Module):
    def __init__(self, input_dim, latent_dim, weight_shapes):
        super(HyperNetwork, self).__init__()
        self.input_dim = input_dim
        self.weight_shapes = weight_shapes

        total_params = sum(shape.numel() for shape in self.weight_shapes.values())

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.PReLU(),
            nn.Dropout(0.1),  # Add dropout
            nn.Linear(16, 32),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, latent_dim),
            nn.PReLU(),
            # nn.Linear(32, total_params)
        )
        # self.mlp_skip_norm = nn.Sequential(
        #     nn.Linear(input_dim, 16),
        #     # nn.LayerNorm(16),  # Layer normalization
        #     nn.PReLU(),
        #     nn.Linear(16, 32),
        #     # nn.LayerNorm(32),  # Layer normalization
        #     nn.PReLU(),
        #     nn.Linear(32, total_params)
        # )
        # self.skip1 = nn.Linear(input_dim, 16)  # Connects input directly to 2nd hidden layer

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1), # 8 16 32
            # nn.Linear(latent_dim, 16),
            nn.PReLU(), # SELU(),
            # SineActivation(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), # 16 32 64
            # nn.Linear(16, 32),
            nn.PReLU(), # SELU(),
            # SineActivation(),
            nn.Flatten(),
            nn.Linear(32 * latent_dim, total_params)
            # nn.Linear(32, total_params)
        )
        # self.cnn3d = nn.Sequential(
        #     nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, padding=1),  # Conv3d layer 1
        #     nn.PReLU(),
        #     nn.Conv3d(16, 32, kernel_size=3, padding=1),  # Conv3d layer 2
        #     nn.PReLU(),
        #     nn.Flatten()
        # )

        # self.apply(sine_init)

        # self.apply(kaiming_init)
        # self.apply(prelu_init)

    def forward(self, x):
        # print(f"Original input shape: {x.shape}")
        x = self.mlp(x).unsqueeze(1)  # Transform and add a channel dimension for CNN
        # x = self.mlp(x) # for another MLP
        # weights_flat = self.mlp(x) # no CNN
        # print(f"After MLP, shape: {x.shape}")
        # x = x.unsqueeze(2).unsqueeze(3)  # Add spatial dimensions for Conv3d
        # weights_flat = self.cnn3d(x)
        weights_flat = self.cnn(x) # with MLP
        # weights_flat = self.cnn(x.unsqueeze(1)) # without MLP
        # x = x.unsqueeze(1).unsqueeze(2).unsqueeze(3)  # Reshape to (B, C, D, H, W) format for Conv3d
        # x = x.unsqueeze(1).unsqueeze(2)  # Reshape to (Batch, Channel, Height, Width) for Conv2d
        # weights_flat = self.cnn3d(x)  # Pass through the Conv3d CNN
        # print(f"After CNN, shape: {weights_flat.shape}")
        return self._reshape_weights(weights_flat)

        # # MLP that includes skip connections
        # out = self.mlp_skip_norm[0:2](x)  # Pass through the first part of MLP
        # skip_out = self.skip1(x)  # Compute skip connection
        # weights_flat = self.mlp_skip_norm[2:](out + skip_out)  # Add skip connection and pass through the rest
        # return self._reshape_weights(weights_flat)

    def _reshape_weights(self, weights_flat):
        weights = {}
        offset = 0
        for name, shape in self.weight_shapes.items():
            size = shape.numel()
            weights[name] = weights_flat[:, offset:offset + size].view(-1, *shape)
            offset += size
        return weights
    
    # def get_flattened_size(self, latent_dim):
    #     # Helper function to calculate the flattened size after Conv3d layers
    #     # Use a dummy input tensor with batch size 1 and shape (B, C, D, H, W)
    #     sample_tensor = torch.zeros(1, 1, latent_dim, latent_dim, latent_dim)
    #     conv_output = self.cnn3d(sample_tensor)  # Pass through Conv3d layers (excluding the Linear)
    #     return conv_output.numel() // sample_tensor.size(0)  # Return the flattened size

# """

refine = False # True
use_deconv = True # True scale = [1, 1, 1, 1] if use_deconv else [4, 2, 2, 1]

class ConvBlock(nn.Module):
    def __init__(self, in_planes, c=64, input_dim=2): # input_dim=3 2
        super(ConvBlock, self).__init__()
        self.in_planes = in_planes
        self.c = c

        # Define the shapes of the weights for the hypernetwork
        self.weight_shapes = {
            # 'conv0_0_weight': torch.Size([c//2, in_planes, 3, 3, 3]), # 4, 4, 4
            'conv0_0_weight': torch.Size([c//2, in_planes, 4, 4, 4]), # 4, 4, 4
            'conv0_0_bias': torch.Size([c//2]),
            # 'conv0_1_weight': torch.Size([c, c//2, 3, 3, 3]), # 4, 4, 4
            'conv0_1_weight': torch.Size([c, c//2, 4, 4, 4]), # 4, 4, 4
            'conv0_1_bias': torch.Size([c]),
            ###
            'convblock_0_weight': torch.Size([c, c, 3, 3, 3]),
            'convblock_0_bias': torch.Size([c]),
            'convblock_1_weight': torch.Size([c, c, 3, 3, 3]),
            'convblock_1_bias': torch.Size([c]),
            'convblock_2_weight': torch.Size([c, c, 3, 3, 3]),
            'convblock_2_bias': torch.Size([c]),
            ###
            # 'convblock_3_weight': torch.Size([c, c, 3, 3, 3]),
            # 'convblock_3_bias': torch.Size([c]),
            # 'convblock_4_weight': torch.Size([c, c, 3, 3, 3]),
            # 'convblock_4_bias': torch.Size([c]),
            # 'convblock_5_weight': torch.Size([c, c, 3, 3, 3]),
            # 'convblock_5_bias': torch.Size([c]),
            # 'convblock_6_weight': torch.Size([c, c, 3, 3, 3]),
            # 'convblock_6_bias': torch.Size([c]),
            # 'deconvblock_0_weight': torch.Size([c, c, 4, 4, 4]),
            # 'deconvblock_0_bias': torch.Size([c]),
            # 'deconvblock_1_weight': torch.Size([c, c//2, 4, 4, 4]),
            # 'deconvblock_1_bias': torch.Size([c//2]),
            # 'deconvblock_2_weight': torch.Size([c//2, 7, 4, 4, 4]),
            # 'deconvblock_2_bias': torch.Size([7]),
            # 'convblock_7_weight': torch.Size([self.c, self.c, 3, 3, 3]),
            # 'convblock_7_bias': torch.Size([self.c]),
            # 'convblock_8_weight': torch.Size([self.c, self.c, 3, 3, 3]),
            # 'convblock_8_bias': torch.Size([self.c]),
            # 'convblock_9_weight': torch.Size([self.c, self.c, 3, 3, 3]),
            # 'convblock_9_bias': torch.Size([self.c]),
        }

        # self.weight_shapes = {
        #     'conv0_0_weight': torch.Size([c//2, in_planes, 4, 4, 4]), # out_channels, in_channels
        #     'conv0_0_bias': torch.Size([c//2]),
        #     'conv0_1_weight': torch.Size([c, c//2, 4, 4, 4]),
        #     'conv0_1_bias': torch.Size([c]),
        #     'convblock_0_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_0_bias': torch.Size([c]),
        #     'convblock_1_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_1_bias': torch.Size([c]),
        #     'convblock_2_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_2_bias': torch.Size([c]),
        #     'convblock_3_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_3_bias': torch.Size([c]),
        #     'convblock_4_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_4_bias': torch.Size([c]),
        #     'convblock_5_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_5_bias': torch.Size([c]),
        #     'convblock_6_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_6_bias': torch.Size([c]),
        #     'deconvblock_0_weight': torch.Size([c, c, 4, 4, 4]),
        #     'deconvblock_0_bias': torch.Size([c]),
        #     'convblock_7_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_7_bias': torch.Size([c]),
        #     'convblock_8_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_8_bias': torch.Size([c]),
        #     'convblock_9_weight': torch.Size([c, c, 3, 3, 3]),
        #     'convblock_9_bias': torch.Size([c]),
        #     'deconvblock_1_weight': torch.Size([c, c//2, 4, 4, 4]), # in_channels, out_channels
        #     'deconvblock_1_bias': torch.Size([c//2]),
        #     'deconvblock_2_weight': torch.Size([c//2, 7, 4, 4, 4]),
        #     'deconvblock_2_bias': torch.Size([7]),
        # }

        # Initialize the hypernetwork
        # self.hypernet = HyperNetwork(input_dim, self.weight_shapes)
        latent_dim = input_dim # 2 3
        self.hypernet = HyperNetwork(input_dim, latent_dim, self.weight_shapes)

        # self.hypernet.apply(sine_init)
        # self.hypernet.apply(first_layer_sine_init)

        # self.hypernet.apply(kaiming_init)
        # self.hypernet.apply(prelu_init)

        # Define the layers to be trained normally (FLINT)
        # self.conv0_0 = nn.Conv3d(in_planes, c//2, 4, stride=2, padding=1)
        # self.conv0_1 = nn.Conv3d(c//2, c, 4, stride=2, padding=1)
        #
        # self.convblock_0 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        # self.convblock_1 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        # self.convblock_2 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_3 = nn.Conv3d(c, c, 3, stride=2, padding=1)
        # self.convblock_3 = nn.Conv3d(c, c, 3, stride=1, padding=1) # test
        self.convblock_4 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_5 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_6 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_7 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_8 = nn.Conv3d(c, c, 3, stride=1, padding=1)
        self.convblock_9 = nn.Conv3d(c, c, 3, stride=1, padding=1)

        # Define the deconvolutional layers to be used in FLINT
        self.deconvblock_0 = nn.ConvTranspose3d(c, c, 4, stride=2, padding=1)
        # self.deconvblock_0 = nn.ConvTranspose3d(c, c, 4, stride=1, padding=1) # test st=1
        # self.deconvblock_0 = nn.ConvTranspose3d(c, c, 3, stride=1, padding=1) # test st=1, kernel 3
        self.deconvblock_1 = nn.ConvTranspose3d(c, c//2, 4, stride=2, padding=1)
        self.deconvblock_2 = nn.ConvTranspose3d(c//2, 7, 4, stride=2, padding=1)

        # Define PReLU activations
        self.prelu = nn.PReLU()
        # self.prelu = nn.SELU()
        # self.prelu = SineActivation()

        # Freeze FLINT layers
        # self.freeze_FLINT_layers()

    # def freeze_FLINT_layers(self):
    #     FLINT_layers = [
    #         self.convblock_0,
    #         self.convblock_1,
    #         self.convblock_2,
    #         self.convblock_3,
    #         self.convblock_4,
    #         self.convblock_5,
    #         self.convblock_6,
    #         self.convblock_7,
    #         self.convblock_8,
    #         self.convblock_9
    #     ]
    #     for layer in FLINT_layers:
    #         for param in layer.parameters():
    #             param.requires_grad = False

    def forward(self, x, flow, scale, params):
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="trilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor=1. / scale, mode="trilinear", align_corners=False) * (1. / scale)
            x = torch.cat((x, flow), 1)

        batch_size = x.size(0)
        # Initialize a list to store the output of each element in the batch
        outputs = []
        # Iterate over each element in the batch
        for i in range(batch_size):
            current_params = params[i].unsqueeze(0)
            weight_dict = self.hypernet(current_params)
            weight_dict = {name: weight_dict[name].squeeze(0) for name in weight_dict}  # Remove the batch dim from weights

            x_i = x[i].unsqueeze(0)

            # with torch.no_grad():
            x_i = functional_conv3d(x_i, weight_dict['conv0_0_weight'], weight_dict['conv0_0_bias'], self.c//2, stride=2, padding=2)
            x_i = functional_conv3d(x_i, weight_dict['conv0_1_weight'], weight_dict['conv0_1_bias'], self.c, stride=2, padding=2)
            # x_i = self.prelu(self.conv0_0(x_i))
            # x_i = self.prelu(self.conv0_1(x_i))
            
            # print(f"x_i after initial convs: requires_grad={x_i.requires_grad}")
            # x_i.requires_grad = True
        
            # hypernetwork-generated weights for the specified layers
            x_i = functional_conv3d(x_i, weight_dict['convblock_0_weight'], weight_dict['convblock_0_bias'], self.c, stride=1, padding=1)
            x_i = functional_conv3d(x_i, weight_dict['convblock_1_weight'], weight_dict['convblock_1_bias'], self.c, stride=1, padding=1)
            x_i = functional_conv3d(x_i, weight_dict['convblock_2_weight'], weight_dict['convblock_2_bias'], self.c, stride=1, padding=1)

            # x_i = self.prelu(self.convblock_0(x_i))
            # x_i = self.prelu(self.convblock_1(x_i))
            # x_i = self.prelu(self.convblock_2(x_i))
            x_i = self.prelu(self.convblock_3(x_i))
            x_i = self.prelu(self.convblock_4(x_i))
            x_i = self.prelu(self.convblock_5(x_i))
            x_i = self.prelu(self.convblock_6(x_i))
            # print(f"x_i after convblock_6: requires_grad={x_i.requires_grad}")

            # x_i = functional_conv3d(x_i, weight_dict['convblock_3_weight'], weight_dict['convblock_3_bias'], self.c, stride=2, padding=1)
            # x_i = functional_conv3d(x_i, weight_dict['convblock_3_weight'], weight_dict['convblock_3_bias'], self.c, stride=1, padding=1) # test
            # x_i = functional_conv3d(x_i, weight_dict['convblock_4_weight'], weight_dict['convblock_4_bias'], self.c, stride=1, padding=1)
            # x_i = functional_conv3d(x_i, weight_dict['convblock_5_weight'], weight_dict['convblock_5_bias'], self.c, stride=1, padding=1)
            # x_i = functional_conv3d(x_i, weight_dict['convblock_6_weight'], weight_dict['convblock_6_bias'], self.c, stride=1, padding=1)

            # with torch.no_grad():
            # x_i = functional_conv_transpose3d(x_i, weight_dict['deconvblock_0_weight'], weight_dict['deconvblock_0_bias'], self.c, stride=2)
            x_i = self.deconvblock_0(x_i) # Use the standard deconv layers
            # print(f"x_i after first deconv: requires_grad={x_i.requires_grad}")
            # x_i.requires_grad = True
            x_i = self.prelu(self.convblock_7(x_i))
            x_i = self.prelu(self.convblock_8(x_i))
            x_i = self.prelu(self.convblock_9(x_i))
            # x_i = functional_conv3d(x_i, weight_dict['convblock_7_weight'], weight_dict['convblock_7_bias'], self.c, stride=1, padding=1)
            # x_i = functional_conv3d(x_i, weight_dict['convblock_8_weight'], weight_dict['convblock_8_bias'], self.c, stride=1, padding=1)
            # x_i = functional_conv3d(x_i, weight_dict['convblock_9_weight'], weight_dict['convblock_9_bias'], self.c, stride=1, padding=1)
            
            # print(f"x_i after convblock_9: requires_grad={x_i.requires_grad}")

            convblock_out = x_i
            convblock_out = convblock_out[:, :, :x_i.shape[2], :x_i.shape[3], :x_i.shape[4]]
            x_i = convblock_out + x_i

            # with torch.no_grad():
            # x_i = functional_conv_transpose3d(x_i, weight_dict['deconvblock_1_weight'], weight_dict['deconvblock_1_bias'], self.c//2)
            # x_i = functional_conv_transpose3d(x_i, weight_dict['deconvblock_2_weight'], weight_dict['deconvblock_2_bias'], 7)
            x_i = self.deconvblock_1(x_i)
            x_i = self.deconvblock_2(x_i)
            # print(f"x_i before final requires_grad setting: requires_grad={x_i.requires_grad}")
            
            # x_i.requires_grad = True

            flow_i = x_i[:, :6]
            mask_i = x_i[:, 6:7]

            outputs.append((flow_i, mask_i))

            # with torch.no_grad():
            # Variable._execution_engine.run_backward()  # Calls into the C++ engine to run the backward pass
            # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

            """
            # Process the current element using the generated weights
            x_i = x[i].unsqueeze(0)  # Add batch dimension back for conv operation

            # print(f"x_i is on device: {x_i.device}")
            # for name, tensor in weight_dict.items():
            #     print(f"{name} is on device: {tensor.device}")

            # Perform the convolutions using the extracted weights
            x_i = functional_conv3d(x_i, weight_dict['conv0_0_weight'], weight_dict['conv0_0_bias'], self.c//2, stride=2, padding=1)
            x_i = functional_conv3d(x_i, weight_dict['conv0_1_weight'], weight_dict['conv0_1_bias'], self.c, stride=2, padding=1)
            # print(f"conv0_1_weight")
            x_i = functional_conv3d(x_i, weight_dict['convblock_0_weight'], weight_dict['convblock_0_bias'], self.c)
            x_i = functional_conv3d(x_i, weight_dict['convblock_1_weight'], weight_dict['convblock_1_bias'], self.c)
            x_i = functional_conv3d(x_i, weight_dict['convblock_2_weight'], weight_dict['convblock_2_bias'], self.c)
            # print(f"convblock_2_weight")
            x_i = functional_conv3d(x_i, weight_dict['convblock_3_weight'], weight_dict['convblock_3_bias'], self.c, stride=2)
            x_i = functional_conv3d(x_i, weight_dict['convblock_4_weight'], weight_dict['convblock_4_bias'], self.c)
            x_i = functional_conv3d(x_i, weight_dict['convblock_5_weight'], weight_dict['convblock_5_bias'], self.c)
            x_i = functional_conv3d(x_i, weight_dict['convblock_6_weight'], weight_dict['convblock_6_bias'], self.c)
            # print(f"convblock_6_weight")
            x_i = functional_conv_transpose3d(x_i, weight_dict['deconvblock_0_weight'], weight_dict['deconvblock_0_bias'], self.c, stride=2)
            x_i = functional_conv3d(x_i, weight_dict['convblock_7_weight'], weight_dict['convblock_7_bias'], self.c)
            x_i = functional_conv3d(x_i, weight_dict['convblock_8_weight'], weight_dict['convblock_8_bias'], self.c)
            x_i = functional_conv3d(x_i, weight_dict['convblock_9_weight'], weight_dict['convblock_9_bias'], self.c)
            print(f"convblock_9_weight")
            convblock_out = x_i
            convblock_out = convblock_out[:, :, :x_i.shape[2], :x_i.shape[3], :x_i.shape[4]]
            x_i = convblock_out + x_i

            x_i = functional_conv_transpose3d(x_i, weight_dict['deconvblock_1_weight'], weight_dict['deconvblock_1_bias'], self.c//2)
            x_i = functional_conv_transpose3d(x_i, weight_dict['deconvblock_2_weight'], weight_dict['deconvblock_2_bias'], 7)
            # print(f"deconvblock_2_weight")

            flow_i = x_i[:, :6]  # Best, since I don't use resize
            mask_i = x_i[:, 6:7]

            outputs.append((flow_i, mask_i))
            """

        # Concatenate the outputs to form the final output tensor
        flows = torch.cat([out[0] for out in outputs], dim=0)
        masks = torch.cat([out[1] for out in outputs], dim=0)

        return flows, masks


class HyperFLINTNet(nn.Module):
    def __init__(self, blocks_num=2, student_teacher=False):
        super(HyperFLINTNet, self).__init__()
        self.blocks_num = blocks_num
        self.student_teacher = student_teacher
        self.refine = refine
        self.use_deconv = use_deconv
        input_dim = 3 # 2 3

        if blocks_num == 1:
            self.block0 = ConvBlock(3, c=128, input_dim=input_dim) # 6 c=240 128 256
        if blocks_num == 2:
            self.block0 = ConvBlock(3, c=128, input_dim=input_dim) # 6 c=240 128 256
            self.block1 = ConvBlock(6+6, c=64, input_dim=input_dim) # 13 c=90 96 64 128
        if blocks_num == 3:
            self.block0 = ConvBlock(3, c=128, input_dim=input_dim) # 6 c=240 128 256
            self.block1 = ConvBlock(6+6, c=96, input_dim=input_dim) # 13 c=150 96 192
            self.block2 = ConvBlock(6+6, c=64, input_dim=input_dim) # 13 c=90 96 64 128
        elif blocks_num == 4:
            self.block0 = ConvBlock(3, c=128, input_dim=input_dim) # 128 256 320 400
            self.block1 = ConvBlock(6+6, c=96, input_dim=input_dim) # 96 192 256 320
            self.block2 = ConvBlock(6+6, c=96, input_dim=input_dim) # 96 192 256 320
            self.block3 = ConvBlock(6+6, c=64, input_dim=input_dim) # 64 128 192 256
        elif blocks_num == 5:
            self.block0 = ConvBlock(3, c=128, input_dim=input_dim) # 128 256 320 400
            self.block1 = ConvBlock(6+6, c=96, input_dim=input_dim) # 96 192 256 320
            self.block2 = ConvBlock(6+6, c=96, input_dim=input_dim) # 96 192 256 320
            self.block3 = ConvBlock(6+6, c=64, input_dim=input_dim) # 64 128 192 256
            self.block4 = ConvBlock(6+6, c=64, input_dim=input_dim) # 64 128 192 256

        if student_teacher:
            self.block_tea = ConvBlock(7+6, c=128, input_dim=3) # 90 128 192 256
        if refine:
            self.contextnet = Contextnet()
            self.unet = Unet()

    # @torch.cuda.amp.autocast(True)
    # def forward(self, x, scale=[1, 1, 1], timestep=0.5): # scale=[4,2,1] scale=[1, 1, 1]
    # def forward(self, x): # scale=[4, 2, 1] scale=[1, 1, 1] t=0.5
    def forward(self, x, params): 
        # img0 = x[:, :1] 
        # img1 = x[:, 1:2]
        # gt = x[:, 2:3] # In inference time, gt is None
        img0 = x[:, :1]
        img1 = x[:, 1:2]
        gt = x[:, 2:3]
        t_mask = x[:, 3:4]

        flow_list = [] # flow_list represents flow predictions by each ConvBlock, 6 channels in 3D, Ft->0, Ft->1
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_distill = 0
        if self.blocks_num == 2:
            stu = [self.block0, self.block1]
            scale = [1, 1] if use_deconv else [2, 1]
        elif self.blocks_num == 3:
            stu = [self.block0, self.block1, self.block2]
            scale = [1, 1, 1] if use_deconv else [4, 2, 1]
        elif self.blocks_num == 4:
            stu = [self.block0, self.block1, self.block2, self.block3]
            # scale = [1, 1, 1, 1] if use_deconv else [8, 4, 2, 1]
            scale = [1, 1, 1, 1] if use_deconv else [4, 2, 2, 1]
        elif self.blocks_num == 5:
            stu = [self.block0, self.block1, self.block2, self.block3, self.block4]
            scale = [1, 1, 1, 1, 1] if use_deconv else [4, 2, 2, 1, 1]
        # stu = [self.block0, self.block1, self.block2]
        # stu = [self.block0, self.block2]
        for i in range(self.blocks_num):
            if flow != None:
                # correct shapes
                max_shape_2 = min(img0.shape[2], warped_img0.shape[2])
                max_shape_3 = min(img0.shape[3], warped_img0.shape[3])
                max_shape_4 = min(img0.shape[4], warped_img0.shape[4])
                # print(max_shape_2, max_shape_3)
                img0 = img0[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                img1 = img1[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                warped_img0 = warped_img0[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                warped_img1 = warped_img1[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                mask = mask[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                flow = flow[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                # gt = gt[:,:,:max_shape_2,:max_shape_3]
                # print(i)
                # print(img0.shape, img1.shape, warped_img0.shape, warped_img1.shape, mask.shape, gt.shape)
                # flow_d, mask_d = stu[i](torch.cat((img0, img1, warped_img0, warped_img1, mask), 1), flow, scale=scale[i])
                flow_d, mask_d = stu[i](torch.cat((img0, img1, t_mask, warped_img0, warped_img1, mask), 1), flow, scale=scale[i], params=params)
                flow_d = flow_d[:,:,:img0.shape[2],:img0.shape[3],:img0.shape[4]]
                mask_d = mask_d[:,:,:img0.shape[2],:img0.shape[3],:img0.shape[4]]
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                # print(i)
                # print("cat", torch.cat((img0, img1), 1).shape)
                # input("flow == None")
                # flow, mask = stu[i](torch.cat((img0, img1), 1), None, scale=scale[i]) # stu[0]
                flow, mask = stu[i](torch.cat((img0, img1, t_mask), 1), None, scale=scale[i], params=params)
                # print(flow, mask) # nan
                # input("flow")
            # correct shapes
            max_shape_2 = min(img0.shape[2], warped_img0.shape[2])
            max_shape_3 = min(img0.shape[3], warped_img0.shape[3])
            max_shape_4 = min(img0.shape[4], warped_img0.shape[4])
            flow = flow[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            mask = mask[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            img0 = img0[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            img1 = img1[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            # print("img0:", img0.shape)
            # print("flow:", flow.shape)
            # input("x")
            warped_img0 = warp(img0, flow[:, :3]) # Ft->0
            warped_img1 = warp(img1, flow[:, 3:6]) # Ft->1
            # print("img0:", img0.shape)
            # print("warped_img0:", warped_img0.shape)
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)

            # print("gt:", gt.shape[1])
            # input("gt")
            
        if self.student_teacher:
            if gt.shape[1] == 1: # 3
                # input("gt")
                # print("gt.shape[1] = 1")
                # print(img0.shape, img1.shape, warped_img0.shape, warped_img1.shape, mask.shape, gt.shape)
                # correct shapes
                max_shape_2 = min(img0.shape[2], warped_img0.shape[2])
                max_shape_3 = min(img0.shape[3], warped_img0.shape[3])
                max_shape_4 = min(img0.shape[4], warped_img0.shape[4])
                img0 = img0[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                img1 = img1[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                warped_img0 = warped_img0[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                warped_img1 = warped_img1[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                mask = mask[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                flow = flow[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                gt = gt[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                # print(img0.shape, img1.shape, warped_img0.shape, warped_img1.shape, mask.shape, gt.shape)
                # print("flow:", flow.shape)

                # """
                # print("before block tea", torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1).shape)
                # flow_d, mask_d = self.block_tea(torch.cat((img0, img1, warped_img0, warped_img1, mask, gt), 1), flow, scale=1)
                flow_d, mask_d = self.block_tea(torch.cat((img0, img1, t_mask, warped_img0, warped_img1, mask, gt), 1), flow, scale=1, params=params)
                flow_d = flow_d[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                mask_d = mask_d[:,:,:max_shape_2,:max_shape_3,:max_shape_4]
                
                flow_teacher = flow + flow_d
                # print("img0 before warp:", img0.shape)
                warped_img0_teacher = warp(img0, flow_teacher[:, :3])
                # print("warped_img0_teacher:", warped_img0_teacher.shape)
                warped_img1_teacher = warp(img1, flow_teacher[:, 3:6])
                mask_teacher = torch.sigmoid(mask + mask_d)
                merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
                # """
                # flow_teacher = 0
                # merged_teacher = 0
            else:
                # print("flow_teacher = None")
                flow_teacher = None
                merged_teacher = None

        for i in range(self.blocks_num):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            merged[i] = merged[i][:,:,:gt.shape[2],:gt.shape[3],:gt.shape[4]]
            
            # """
            if self.student_teacher:
                if gt.shape[1] == 1: # 3
                    flow_list[i] = flow_list[i][:,:,:flow_teacher.shape[2],:flow_teacher.shape[3],:flow_teacher.shape[4]]

                    loss_mask = ( (merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01 ).float().detach()
                    # this is 0 tensor
                    # loss_distill += ((flow_teacher.detach() - flow_list[i]).abs() * loss_mask).mean()
                    # print(loss_distill)
                    # print(flow_teacher.detach())
                    # print(flow_list[i])
                    # print(np.mean(loss_mask.detach().cpu().numpy()))
                    # input("loss_mask")
                    loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
                    # loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5).mean()
                    # print( (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5).mean() )
                    # print(loss_mask.mean())
                    # input("loss_distill")
            # """

        # print("in FLINTNet, forward")
        if refine:
            # c0 = self.contextnet(img0, flow[:, :2])
            # c1 = self.contextnet(img1, flow[:, 2:4])
            # tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
            # res = tmp[:, :3] * 2 - 1
            # merged[2] = torch.clamp(merged[2] + res, 0, 1)
            print("refined")
        # return flow_list, mask_list, merged, flow_teacher, merged_teacher, loss_distill # mask_list[2]
        if self.student_teacher:
            return flow_list, mask_list, merged, flow_teacher, merged_teacher, loss_distill
            # return flow_list, mask_list, merged, warped_img0, flow_teacher, merged_teacher, loss_distill
        else:
            return flow_list, mask_list, merged
