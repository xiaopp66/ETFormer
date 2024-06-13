import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ptflops import get_model_complexity_info
# from ETFormer.network_architecture.TC_CoNet.TCCoNet_acdc import TCCoNet
from ETFormer.network_architecture.EA_ds import UNETR_PP
from ETFormer.network_architecture.ETFormer_acdc import ETFormer
from torch import nn

# ACDC
# model = TCCoNet(crop_size=[14, 160, 160],
#                  embedding_dim=96,
#                  input_channels=1,
#                  num_classes=4,
#                  conv_op=nn.Conv3d,
#                  depths=[2, 2, 2, 2],
#                  num_heads=[3, 6, 12, 24],
#                  patch_size=[1, 4, 4],
#                  window_size=[[3, 5, 5], [3, 5, 5], [7, 10, 10], [3, 5, 5]],
#                  down_stride=[[1, 4, 4], [1, 8, 8], [2, 16, 16], [4, 32, 32]],
#                  deep_supervision=False)
# tumor,150.94 GMac params:  86.79 M
#ACDC
model = ETFormer(crop_size=[64, 128, 128],
                 embedding_dim=192,
                 input_channels=1,
                 num_classes=4,
                 conv_op=nn.Conv3d,
                 depths=[2, 2, 2, 2],
                 num_heads=[6, 12, 24, 48],
                 patch_size=[2, 4, 4],
                 window_size=[4, 4, 8, 4],
                 down_stride=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
                 deep_supervision=True)
# ACDC
# model = UNETR_PP(in_channels=1,
#                  out_channels=9,
#                  # img_size=[64, 128, 128],
#                  feature_size=16,
#                  num_heads=4,
#                  depths=[3, 3, 3, 3],
#                  dims=[32, 64, 128, 256],
#                  do_ds=True,
#                  )
#

flops, params = get_model_complexity_info(model, (1, 64, 128, 128), as_strings=True, print_per_layer_stat=True)
print('flops: ', flops, 'params: ', params)
