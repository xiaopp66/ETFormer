ETFormer

Environment
Please prepare an environment with Python 3.7, Pytorch 1.7.1, and Windows 10.

Dataset Preparation
Datasets can be acquired via following links:

Dataset I ACDC

Dataset II The Synapse multi-organ CT dataset

Dataset III Brain_tumor

Preprocess Data
TCCoNet_convert_decathlon_task -i D:\Codes\Medical_image\UploadGitHub\TCCoNet\DATASET\TCCoNet_raw\TCCoNet_raw_data
TCCoNet_plan_and_preprocess -t 2
Functions of scripts
Network architecture:
TCCoNet\TCCoNet\network_architecture\TCCoNet_acdc.py
TCCoNet\TCCoNet\network_architecture\TCCoNet_synapse.py
TCCoNet\TCCoNet\network_architecture\TCCoNet_tumor.py
TCCoNet\TCCoNet\network_architecture\TCCoNet_heart.py
TCCoNet\TCCoNet\network_architecture\TCCoNet_lung.py
Trainer for dataset:
TCCoNet\TCCoNet\training\network_training\TCCoNetTrainerV2_TCCoNet_acdc.py
TCCoNet\TCCoNet\training\network_training\TCCoNetTrainerV2_TCCoNet_synapse.py
TCCoNet\TCCoNet\training\network_training\TCCoNetTrainerV2_TCCoNet_tumor.py
TCCoNet\TCCoNet\training\network_training\TCCoNetTrainerV2_TCCoNet_heart.py
TCCoNet\TCCoNet\training\network_training\TCCoNetTrainerV2_TCCoNet_lung.py
Train Model
python run_training.py 3d_fullres TCCoNetTrainerV2_TCCoNet_synapse 2 0
Test Model
python predict.py -i D:\Codes\Medical_image\UploadGitHub\TCCoNet\DATASET\TCCoNet_raw\TCCoNet_raw_data\Task002_Synapse\imagesTs -o D:\Codes\Medical_image\UploadGitHub\TCCoNet\DATASET\TCCoNet_raw\TCCoNet_raw_data\Task002_Synapse\imagesTs_infer -m D:\Codes\Medical_image\UploadGitHub\TCCoNet\DATASET\TCCoNet_trained_models\TCCoNet\3d_fullres\Task002_Synapse\TCCoNetTrainerV2_TCCoNet_synapse__TCCoNetPlansv2.1 -f 0

python TCCoNet/inference_synapse.py

Acknowledgements
This repository makes liberal use of code from Swin Transformer, nnUNet.
