import torch
import torch.nn as nn
import numpy as np
from models.modules.embedder import get_embedder
from models.modules.triplane import Hash_triplane, Hash_grid

class SDFNetwork(nn.Module):
    def __init__(self,
                 point_size,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 use_plane_feature=False,
                 use_grid_feature=False,
                 use_point_transformer=False,
                 point_transformer_grid_size=0.01,
                 inside_outside=False):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        base_feat_dim = dims[0]

        self.coord_dim = d_in
        self.use_point_transformer = use_point_transformer
        self.point_transformer_grid_size = point_transformer_grid_size
        self.point_transformer_out_dim = 0
        if self.use_point_transformer:
            try:
                from models.modules.PointTransformerv3 import PointTransformerV3
            except ImportError as err:
                raise ImportError(
                    "PointTransformerV3 requires spconv/flash-attn dependencies. "
                    "Install them or set use_point_transformer=False."
                ) from err

            pt_in_channels = base_feat_dim + d_in
            self.point_transformer = PointTransformerV3(in_channels=pt_in_channels)
            self.point_transformer_out_dim = 64
            dims[0] += self.point_transformer_out_dim

        self.use_grid_feature = use_grid_feature
        if self.use_grid_feature:
            self.grid_encoding = Hash_grid(point_size=point_size, use_pro=True)

        self.use_plane_feature = use_plane_feature
        if self.use_plane_feature:
            self.plane_encoding = Hash_triplane(point_size=point_size, multires=multires, use_pro=True)

        if self.use_grid_feature or self.use_plane_feature:
            dims[0] += 16*2

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
            
        self.activation = nn.ReLU()

    def forward(self, inputs, step):
        scaled_inputs = inputs * self.scale

        if self.embed_fn_fine is not None:
            embedded_inputs = self.embed_fn_fine(scaled_inputs)
        else:
            embedded_inputs = scaled_inputs

        if self.use_point_transformer:
            pt_coord = scaled_inputs.detach()
            pt_feats_in = torch.cat((pt_coord, embedded_inputs.detach()), dim=-1)
            point_dict = {
                'coord': pt_coord,
                'grid_size': self.point_transformer_grid_size,
                'offset': scaled_inputs.new_tensor([scaled_inputs.shape[0]], dtype=torch.long),
                'feat': pt_feats_in,
            }
            transformer_out = self.point_transformer(point_dict)
            embedded_inputs = torch.cat((embedded_inputs, transformer_out['feat']), dim=-1)

        feature = 0.
        if self.use_plane_feature:
            feature += self.plane_encoding(scaled_inputs, step)

        if self.use_grid_feature:
            feature += self.grid_encoding(scaled_inputs, step)

        if self.use_plane_feature or self.use_grid_feature:
            embedded_inputs = torch.cat((embedded_inputs, feature), dim=-1)

        x = embedded_inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, embedded_inputs], 1) / np.sqrt(2)

            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)

        return x / self.scale

    def sdf(self, x, step):
        return self.forward(x, step)

    def gradient(self, x, step):
        x.requires_grad_(True)
        y = self.sdf(x,step)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients, y


