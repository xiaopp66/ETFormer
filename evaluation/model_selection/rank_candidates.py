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


import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from ETFormer.paths import network_training_output_dir

if __name__ == "__main__":
    # run collect_all_fold0_results_and_summarize_in_one_csv.py first
    summary_files_dir = join(network_training_output_dir, "summary_jsons_fold0_new")
    output_file = join(network_training_output_dir, "summary.csv")

    folds = (0, )
    folds_str = ""
    for f in folds:
        folds_str += str(f)

    plans = "ETFormerPlans"

    overwrite_plans = {
        'ETFormerTrainerV2_2': ["ETFormerPlans", "ETFormerPlansisoPatchesInVoxels"], # r
        'ETFormerTrainerV2': ["ETFormerPlansnonCT", "ETFormerPlansCT2", "ETFormerPlansallConv3x3",
                            "ETFormerPlansfixedisoPatchesInVoxels", "ETFormerPlanstargetSpacingForAnisoAxis",
                            "ETFormerPlanspoolBasedOnSpacing", "ETFormerPlansfixedisoPatchesInmm", "ETFormerPlansv2.1"],
        'ETFormerTrainerV2_warmup': ["ETFormerPlans", "ETFormerPlansv2.1", "ETFormerPlansv2.1_big", "ETFormerPlansv2.1_verybig"],
        'ETFormerTrainerV2_cycleAtEnd': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_cycleAtEnd2': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_reduceMomentumDuringTraining': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_graduallyTransitionFromCEToDice': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_independentScalePerAxis': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_Mish': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_Ranger_lr3en4': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_fp32': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_GN': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_momentum098': ["ETFormerPlans", "ETFormerPlansv2.1"],
        'ETFormerTrainerV2_momentum09': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_DP': ["ETFormerPlansv2.1_verybig"],
        'ETFormerTrainerV2_DDP': ["ETFormerPlansv2.1_verybig"],
        'ETFormerTrainerV2_FRN': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_resample33': ["ETFormerPlansv2.3"],
        'ETFormerTrainerV2_O2': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_ResencUNet': ["ETFormerPlans_FabiansResUNet_v2.1"],
        'ETFormerTrainerV2_DA2': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_allConv3x3': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_ForceBD': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_ForceSD': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_LReLU_slope_2en1': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_lReLU_convReLUIN': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_ReLU': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_ReLU_biasInSegOutput': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_ReLU_convReLUIN': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_lReLU_biasInSegOutput': ["ETFormerPlansv2.1"],
        #'ETFormerTrainerV2_Loss_MCC': ["ETFormerPlansv2.1"],
        #'ETFormerTrainerV2_Loss_MCCnoBG': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_Loss_DicewithBG': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_Loss_Dice_LR1en3': ["ETFormerPlansv2.1"],
        'ETFormerTrainerV2_Loss_Dice': ["ETFormerPlans", "ETFormerPlansv2.1"],
        'ETFormerTrainerV2_Loss_DicewithBG_LR1en3': ["ETFormerPlansv2.1"],
        # 'ETFormerTrainerV2_fp32': ["ETFormerPlansv2.1"],
        # 'ETFormerTrainerV2_fp32': ["ETFormerPlansv2.1"],
        # 'ETFormerTrainerV2_fp32': ["ETFormerPlansv2.1"],
        # 'ETFormerTrainerV2_fp32': ["ETFormerPlansv2.1"],
        # 'ETFormerTrainerV2_fp32': ["ETFormerPlansv2.1"],

    }

    trainers = ['ETFormerTrainer'] + ['ETFormerTrainerNewCandidate%d' % i for i in range(1, 28)] + [
        'ETFormerTrainerNewCandidate24_2',
        'ETFormerTrainerNewCandidate24_3',
        'ETFormerTrainerNewCandidate26_2',
        'ETFormerTrainerNewCandidate27_2',
        'ETFormerTrainerNewCandidate23_always3DDA',
        'ETFormerTrainerNewCandidate23_corrInit',
        'ETFormerTrainerNewCandidate23_noOversampling',
        'ETFormerTrainerNewCandidate23_softDS',
        'ETFormerTrainerNewCandidate23_softDS2',
        'ETFormerTrainerNewCandidate23_softDS3',
        'ETFormerTrainerNewCandidate23_softDS4',
        'ETFormerTrainerNewCandidate23_2_fp16',
        'ETFormerTrainerNewCandidate23_2',
        'ETFormerTrainerVer2',
        'ETFormerTrainerV2_2',
        'ETFormerTrainerV2_3',
        'ETFormerTrainerV2_3_CE_GDL',
        'ETFormerTrainerV2_3_dcTopk10',
        'ETFormerTrainerV2_3_dcTopk20',
        'ETFormerTrainerV2_3_fp16',
        'ETFormerTrainerV2_3_softDS4',
        'ETFormerTrainerV2_3_softDS4_clean',
        'ETFormerTrainerV2_3_softDS4_clean_improvedDA',
        'ETFormerTrainerV2_3_softDS4_clean_improvedDA_newElDef',
        'ETFormerTrainerV2_3_softDS4_radam',
        'ETFormerTrainerV2_3_softDS4_radam_lowerLR',

        'ETFormerTrainerV2_2_schedule',
        'ETFormerTrainerV2_2_schedule2',
        'ETFormerTrainerV2_2_clean',
        'ETFormerTrainerV2_2_clean_improvedDA_newElDef',

        'ETFormerTrainerV2_2_fixes', # running
        'ETFormerTrainerV2_BN', # running
        'ETFormerTrainerV2_noDeepSupervision', # running
        'ETFormerTrainerV2_softDeepSupervision', # running
        'ETFormerTrainerV2_noDataAugmentation', # running
        'ETFormerTrainerV2_Loss_CE', # running
        'ETFormerTrainerV2_Loss_CEGDL',
        'ETFormerTrainerV2_Loss_Dice',
        'ETFormerTrainerV2_Loss_DiceTopK10',
        'ETFormerTrainerV2_Loss_TopK10',
        'ETFormerTrainerV2_Adam', # running
        'ETFormerTrainerV2_Adam_ETFormerTrainerlr', # running
        'ETFormerTrainerV2_SGD_ReduceOnPlateau', # running
        'ETFormerTrainerV2_SGD_lr1en1', # running
        'ETFormerTrainerV2_SGD_lr1en3', # running
        'ETFormerTrainerV2_fixedNonlin', # running
        'ETFormerTrainerV2_GeLU', # running
        'ETFormerTrainerV2_3ConvPerStage',
        'ETFormerTrainerV2_NoNormalization',
        'ETFormerTrainerV2_Adam_ReduceOnPlateau',
        'ETFormerTrainerV2_fp16',
        'ETFormerTrainerV2', # see overwrite_plans
        'ETFormerTrainerV2_noMirroring',
        'ETFormerTrainerV2_momentum09',
        'ETFormerTrainerV2_momentum095',
        'ETFormerTrainerV2_momentum098',
        'ETFormerTrainerV2_warmup',
        'ETFormerTrainerV2_Loss_Dice_LR1en3',
        'ETFormerTrainerV2_NoNormalization_lr1en3',
        'ETFormerTrainerV2_Loss_Dice_squared',
        'ETFormerTrainerV2_newElDef',
        'ETFormerTrainerV2_fp32',
        'ETFormerTrainerV2_cycleAtEnd',
        'ETFormerTrainerV2_reduceMomentumDuringTraining',
        'ETFormerTrainerV2_graduallyTransitionFromCEToDice',
        'ETFormerTrainerV2_insaneDA',
        'ETFormerTrainerV2_independentScalePerAxis',
        'ETFormerTrainerV2_Mish',
        'ETFormerTrainerV2_Ranger_lr3en4',
        'ETFormerTrainerV2_cycleAtEnd2',
        'ETFormerTrainerV2_GN',
        'ETFormerTrainerV2_DP',
        'ETFormerTrainerV2_FRN',
        'ETFormerTrainerV2_resample33',
        'ETFormerTrainerV2_O2',
        'ETFormerTrainerV2_ResencUNet',
        'ETFormerTrainerV2_DA2',
        'ETFormerTrainerV2_allConv3x3',
        'ETFormerTrainerV2_ForceBD',
        'ETFormerTrainerV2_ForceSD',
        'ETFormerTrainerV2_ReLU',
        'ETFormerTrainerV2_LReLU_slope_2en1',
        'ETFormerTrainerV2_lReLU_convReLUIN',
        'ETFormerTrainerV2_ReLU_biasInSegOutput',
        'ETFormerTrainerV2_ReLU_convReLUIN',
        'ETFormerTrainerV2_lReLU_biasInSegOutput',
        'ETFormerTrainerV2_Loss_DicewithBG_LR1en3',
        #'ETFormerTrainerV2_Loss_MCCnoBG',
        'ETFormerTrainerV2_Loss_DicewithBG',
        # 'ETFormerTrainerV2_Loss_Dice_LR1en3',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
        # 'ETFormerTrainerV2_Ranger_lr3en4',
    ]

    datasets = \
        {"Task001_BrainTumour": ("3d_fullres", ),
        "Task002_Heart": ("3d_fullres",),
        #"Task024_Promise": ("3d_fullres",),
        #"Task027_ACDC": ("3d_fullres",),
        "Task003_Liver": ("3d_fullres", "3d_lowres"),
        "Task004_Hippocampus": ("3d_fullres",),
        "Task005_Prostate": ("3d_fullres",),
        "Task006_Lung": ("3d_fullres", "3d_lowres"),
        "Task007_Pancreas": ("3d_fullres", "3d_lowres"),
        "Task008_HepaticVessel": ("3d_fullres", "3d_lowres"),
        "Task009_Spleen": ("3d_fullres", "3d_lowres"),
        "Task010_Colon": ("3d_fullres", "3d_lowres"),}

    expected_validation_folder = "validation_raw"
    alternative_validation_folder = "validation"
    alternative_alternative_validation_folder = "validation_tiledTrue_doMirror_True"

    interested_in = "mean"

    result_per_dataset = {}
    for d in datasets:
        result_per_dataset[d] = {}
        for c in datasets[d]:
            result_per_dataset[d][c] = []

    valid_trainers = []
    all_trainers = []

    with open(output_file, 'w') as f:
        f.write("trainer,")
        for t in datasets.keys():
            s = t[4:7]
            for c in datasets[t]:
                s1 = s + "_" + c[3]
                f.write("%s," % s1)
        f.write("\n")

        for trainer in trainers:
            trainer_plans = [plans]
            if trainer in overwrite_plans.keys():
                trainer_plans = overwrite_plans[trainer]

            result_per_dataset_here = {}
            for d in datasets:
                result_per_dataset_here[d] = {}

            for p in trainer_plans:
                name = "%s__%s" % (trainer, p)
                all_present = True
                all_trainers.append(name)

                f.write("%s," % name)
                for dataset in datasets.keys():
                    for configuration in datasets[dataset]:
                        summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, expected_validation_folder, folds_str))
                        if not isfile(summary_file):
                            summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (dataset, configuration, trainer, p, alternative_validation_folder, folds_str))
                            if not isfile(summary_file):
                                summary_file = join(summary_files_dir, "%s__%s__%s__%s__%s__%s.json" % (
                                dataset, configuration, trainer, p, alternative_alternative_validation_folder, folds_str))
                                if not isfile(summary_file):
                                    all_present = False
                                    print(name, dataset, configuration, "has missing summary file")
                        if isfile(summary_file):
                            result = load_json(summary_file)['results'][interested_in]['mean']['Dice']
                            result_per_dataset_here[dataset][configuration] = result
                            f.write("%02.4f," % result)
                        else:
                            f.write("NA,")
                            result_per_dataset_here[dataset][configuration] = 0

                f.write("\n")

                if True:
                    valid_trainers.append(name)
                    for d in datasets:
                        for c in datasets[d]:
                            result_per_dataset[d][c].append(result_per_dataset_here[d][c])

    invalid_trainers = [i for i in all_trainers if i not in valid_trainers]

    num_valid = len(valid_trainers)
    num_datasets = len(datasets.keys())
    # create an array that is trainer x dataset. If more than one configuration is there then use the best metric across the two
    all_res = np.zeros((num_valid, num_datasets))
    for j, d in enumerate(datasets.keys()):
        ks = list(result_per_dataset[d].keys())
        tmp = result_per_dataset[d][ks[0]]
        for k in ks[1:]:
            for i in range(len(tmp)):
                tmp[i] = max(tmp[i], result_per_dataset[d][k][i])
        all_res[:, j] = tmp

    ranks_arr = np.zeros_like(all_res)
    for d in range(ranks_arr.shape[1]):
        temp = np.argsort(all_res[:, d])[::-1] # inverse because we want the highest dice to be rank0
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        ranks_arr[:, d] = ranks

    mn = np.mean(ranks_arr, 1)
    for i in np.argsort(mn):
        print(mn[i], valid_trainers[i])

    print()
    print(valid_trainers[np.argmin(mn)])
