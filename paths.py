#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

# do not modify these unless you know what you are doing
my_output_identifier = "ETFormer"
default_plans_identifier = "ETFormerPlansv2.1"
default_data_identifier = 'ETFormerData_plans_v2.1'
default_trainer = "ETFormerTrainerV2_ETFormer_acdc"
default_cascade_trainer = "ETFormerTrainerV2CascadeFullRes"

"""
PLEASE READ paths.md FOR INFORMATION TO HOW TO SET THIS UP
"""

base = r'E:/PGN/ETFormer/DATASET/ETFormer_raw'
preprocessing_output_dir = r'E:/PGN/ETFormer/DATASET/ETFormer_preprocessed'
network_training_output_dir_base = r'E:/PGN/ETFormer/DATASET/ETFormer_trained_models'

if base is not None:
    ETFormer_raw_data = join(base, "ETFormer_raw_data")
    ETFormer_cropped_data = join(base, "ETFormer_cropped_data")
    maybe_mkdir_p(ETFormer_raw_data)
    maybe_mkdir_p(ETFormer_cropped_data)
else:
    print("ETFormer_raw_data_base is not defined and nnU-Net can only be used on data for which preprocessed files "
          "are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like "
          "this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up properly.")
    ETFormer_cropped_data = ETFormer_raw_data = None

if preprocessing_output_dir is not None:
    maybe_mkdir_p(preprocessing_output_dir)
else:
    print("ETFormer_preprocessed is not defined and nnU-Net can not be used for preprocessing "
          "or training. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up.")
    preprocessing_output_dir = None

if network_training_output_dir_base is not None:
    network_training_output_dir = join(network_training_output_dir_base, my_output_identifier)
    maybe_mkdir_p(network_training_output_dir)
else:
    print("RESULTS_FOLDER is not defined and nnU-Net cannot be used for training or "
          "inference. If this is not intended behavior, please read documentation/setting_up_paths.md for information on how to set this "
          "up.")
    network_training_output_dir = None
