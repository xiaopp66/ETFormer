from torch import nn
import torch
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from ETFormer.network_architecture.layers import LayerNorm
from ETFormer.network_architecture.dynunet_block import get_conv_layer, UnetOutBlock, UnetResBlock
from ETFormer.network_architecture.neural_network import SegmentationNetwork

einops, _ = optional_import("einops")

class EfficiertTransformerBlock(nn.Module):
  
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            proj_size: int,
            num_heads: int,
            dropout_rate: float = 0.0,
            pos_embed=False,
    ) -> None:
        """
        Args:
            input_size: the size of the input for each stage.
            hidden_size: dimension of hidden layer.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of attention heads.
            dropout_rate: faction of the input units to drop.
            pos_embed: bool argument to determine if positional embedding is used.

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            print("Hidden size is ", hidden_size)
            print("Num heads is ", num_heads)
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.norm = nn.LayerNorm(hidden_size)
        self.gamma = nn.Parameter(1e-6 * torch.ones(hidden_size), requires_grad=True)
        self.mea_block = Mul_External_Attention(input_size=input_size, dim=hidden_size, proj_size=proj_size,
                                                num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.)
        self.conv51 = UnetResBlock(3, hidden_size, hidden_size, kernel_size=3, stride=1, norm_name="batch")
        self.conv8 = nn.Sequential(nn.Dropout3d(0.1, False), nn.Conv3d(hidden_size, hidden_size, 1))

        self.pos_embed = None
        if pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size))

    def forward(self, x):
        B, C, H, W, D = x.shape

        x = x.reshape(B, C, H * W * D).permute(0, 2, 1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        attn = x + self.gamma * self.mea_block(self.norm(x))

        attn_skip = attn.reshape(B, H, W, D, C).permute(0, 4, 1, 2, 3)  # (B, C, H, W, D)
        attn = self.conv51(attn_skip)
        x = attn_skip + self.conv8(attn)

        return x

class Mul_External_Attention(nn.Module):
    def __init__(self, input_size, dim, proj_size, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads   
        assert dim % num_heads == 0
        self.trans_dims = nn.Linear(dim, dim,bias=False)  

        # 交互q值的映射
        self.E = nn.Linear(dim, dim, bias=qkv_bias)

        #CA
        self.linear_0 = nn.Linear(int(dim//self.num_heads),int(dim*2))
        self.linear_1 = nn.Linear(int(dim*2),int(dim//self.num_heads))

        # SA
        self.linear_2 = nn.Linear(input_size, proj_size)
        self.linear_3 = nn.Linear(proj_size, input_size)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, int(dim//2))  
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        B, N, C = x.shape

        x = self.trans_dims(x)  
        x_s=x
        x = x.view(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)  

        attn = self.linear_0(x)   

        #CA
        attn_CA = attn.softmax(dim=-2)
        attn_CA = attn_CA / (1e-9 +attn_CA.sum(dim=-1, keepdim=True)) 
        attn_CA = self.attn_drop(attn_CA)
        x_CA= self.linear_1(attn_CA).permute(0, 2, 1, 3).reshape(B, N, -1)  

        x_CA = self.proj(x_CA)  
        x_CA = self.proj_drop(x_CA)

        # 交互的q
        q_s = self.E(x_s) 
        q_s = q_s.view(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 3, 1) 

        #SA
        attn_SA = self.linear_2(q_s) 
        attn_SA = attn_SA.softmax(dim=-2)
        attn_SA =attn_SA / (1e-9 + attn_SA.sum(dim=-1, keepdim=True))
        attn_SA = self.attn_drop(attn_SA)
        x_SA= self.linear_3(attn_SA).permute(0, 2, 3, 1).reshape(B, N, -1) 

        x_SA = self.proj(x_SA)  
        x_SA = self.proj_drop(x_SA)

        # 输出拼接
        out = torch.cat((x_CA, x_SA), dim=2)

        return out


class PPCD(nn.Module):
    def __init__(self,dim,atrous_rate):
        super().__init__()

        self.dc_block = nn.Sequential(
            nn.Conv3d(dim,dim,kernel_size=3,padding=atrous_rate,dilation=atrous_rate,bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(dim),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool3d(2)
        self.cbr = nn.Sequential(
            nn.Conv3d(int(dim*2), int(dim*2), kernel_size=3,padding=atrous_rate,dilation=atrous_rate, bias=False),
            nn.BatchNorm3d(int(dim*2)),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):

        res = self.maxpool(x)
        x = self.dc_block(x)
        x_s = torch.cat((x,res),dim=1)
        out = self.cbr(x_s)

        return out

class Encoder(nn.Module):
    def __init__(self, input_size=[32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4], dims=[32, 64, 128, 256],
                 proj_size=[64, 64, 64, 32], depths=[3, 3, 3, 3], num_heads=4, spatial_dims=3, in_channels=1,
                 dropout=0.0, transformer_dropout_rate=0.15, atrous_rate=[2,4,6],**kwargs):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)
        for i in range(3):
            downsample_layer = nn.Sequential(
                PPCD(dims[i],atrous_rate=atrous_rate[i]),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    EfficiertTransformerBlock(input_size=input_size[i], hidden_size=dims[i], proj_size=proj_size[i],
                                     num_heads=num_heads,
                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []

        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class UnetrUpBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(EfficiertTransformerBlock(input_size=out_size, hidden_size=out_channels, proj_size=proj_size,
                                                     num_heads=num_heads,
                                                     dropout_rate=0.15, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        out = out + skip
        out = self.decoder_block[0](out)

        return out


class ETFormer(SegmentationNetwork):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [64, 128, 128],
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (2, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.etformer_encoder = Encoder(dims=dims, depths=depths, num_heads=num_heads)

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=8 * 8 * 8,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=16 * 16 * 16,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=32 * 32 * 32,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            out_size=64 * 128 * 128,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.up1 = nn.ConvTranspose3d(feature_size * 2, out_channels, (1, 2, 2), (1, 2, 2))
            self.up2 = nn.ConvTranspose3d(feature_size * 4, out_channels, (1, 2, 2), (1, 2, 2))

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.reshape(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.etformer_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.up1(dec1), self.up2(dec2)]
        else:
            logits = self.out1(out)

        return logits
