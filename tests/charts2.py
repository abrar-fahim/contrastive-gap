from matplotlib import pyplot as plt


import numpy as np

import ast
import torch




from run_summaries import runs

names = ['L_{CLIP}','L_{CLIP+Uniform+Align}', 'L_{CLIP+Uniform+Align+XUniform}']
colors = ['tab:blue', 'tab:green', 'tab:red']
# plot bar chart of average of image and text variances

# fix plt size
# plt.rcParams['figure.figsize'] = [8, 8] # for pca expaleined varszx
plt.rcParams['figure.figsize'] = [10, 8] # for dim vs acc


datas = [
   {'checkpoint_path': 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'mean_cosine_similarity': 0.591004490852356, 'linear_seperability_accuracy': 1.0, 'centroid_euclidean_distance': 0.6059186458587646, 'val_image_classification_acc': {1: 0.27710600564288596, 3: 0.47339782345828296, 5: 0.5711406690850463, 10: 0.7061668681983071}, 'get_val_image_retrieval_acc': {1: 0.2750906892382104, 3: 0.4723901652559452, 5: 0.5707376058041113, 10: 0.6950826279725917}, 'image_variances': torch.torch.tensor([0.0916, 0.0776, 0.0626, 0.0563, 0.0453, 0.0382, 0.0316, 0.0289, 0.0283,
        0.0260, 0.0247, 0.0233, 0.0225, 0.0204, 0.0200, 0.0184, 0.0173, 0.0162,
        0.0160, 0.0142, 0.0136, 0.0132, 0.0131, 0.0122, 0.0113, 0.0109, 0.0096,
        0.0091, 0.0082, 0.0079, 0.0073, 0.0071, 0.0069, 0.0067, 0.0064, 0.0063,
        0.0060, 0.0059, 0.0054, 0.0052, 0.0050, 0.0048, 0.0045, 0.0045, 0.0042,
        0.0041, 0.0039, 0.0038, 0.0036, 0.0035, 0.0034, 0.0033, 0.0032, 0.0031,
        0.0030, 0.0029, 0.0028, 0.0027, 0.0026, 0.0026, 0.0024, 0.0023, 0.0023,
        0.0022, 0.0021, 0.0021, 0.0020, 0.0020, 0.0019, 0.0019, 0.0018, 0.0018,
        0.0017, 0.0017, 0.0016, 0.0016, 0.0016, 0.0015, 0.0015, 0.0014, 0.0014,
        0.0013, 0.0013, 0.0013, 0.0012, 0.0012, 0.0012, 0.0011, 0.0011, 0.0011,
        0.0011, 0.0011, 0.0010, 0.0010, 0.0010, 0.0010, 0.0009, 0.0009, 0.0009,
        0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008, 0.0007, 0.0007,
        0.0007, 0.0007, 0.0007, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006,
        0.0006, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0004, 0.0004,
        0.0004, 0.0003]), 'text_variances': torch.torch.tensor([0.0990, 0.0755, 0.0599, 0.0579, 0.0418, 0.0390, 0.0322, 0.0295, 0.0291,
        0.0260, 0.0251, 0.0242, 0.0220, 0.0208, 0.0202, 0.0196, 0.0190, 0.0166,
        0.0157, 0.0148, 0.0140, 0.0134, 0.0125, 0.0119, 0.0112, 0.0107, 0.0097,
        0.0093, 0.0090, 0.0088, 0.0083, 0.0077, 0.0074, 0.0068, 0.0066, 0.0065,
        0.0064, 0.0059, 0.0057, 0.0053, 0.0050, 0.0049, 0.0048, 0.0046, 0.0043,
        0.0042, 0.0040, 0.0039, 0.0037, 0.0035, 0.0033, 0.0033, 0.0031, 0.0030,
        0.0029, 0.0028, 0.0027, 0.0026, 0.0025, 0.0024, 0.0023, 0.0022, 0.0021,
        0.0020, 0.0020, 0.0019, 0.0019, 0.0018, 0.0017, 0.0017, 0.0016, 0.0016,
        0.0015, 0.0015, 0.0015, 0.0014, 0.0014, 0.0013, 0.0013, 0.0012, 0.0011,
        0.0011, 0.0011, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010, 0.0009, 0.0009,
        0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0007, 0.0007, 0.0007, 0.0007,
        0.0007, 0.0007, 0.0007, 0.0006, 0.0006, 0.0006, 0.0006, 0.0005, 0.0005,
        0.0005, 0.0005, 0.0005, 0.0005, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004,
        0.0004, 0.0004, 0.0004, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0002, 0.0002]), 'uniformity_loss': {'image_uniformity_loss': torch.torch.tensor(-1.9840, device='cpu'), 'text_uniformity_loss': torch.torch.tensor(-2.0507, device='cpu'), 'total_uniformity_loss': torch.torch.tensor(-2.0173, device='cpu'), 'cross_encoder_uniform_loss': torch.torch.tensor(-2.7869, device='cpu')}, 'alignment_loss': torch.torch.tensor(0.8180, device='cpu')}}


,
    {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt', 'gap_stuff': {'mean_cosine_similarity': 0.7320742011070251, 'linear_seperability_accuracy': 0.7316213494461229, 'centroid_euclidean_distance': 0.07930086553096771, 'val_image_classification_acc': {1: 0.25372833534864975, 3: 0.4347037484885127, 5: 0.5390971382507054, 10: 0.6789600967351874}, 'get_val_image_retrieval_acc': {1: 0.2470777912132205, 3: 0.44095122934300685, 5: 0.5390971382507054, 10: 0.6596130592503023}, 'image_variances': torch.torch.tensor([0.0611, 0.0589, 0.0527, 0.0500, 0.0440, 0.0427, 0.0374, 0.0361, 0.0332,
        0.0306, 0.0293, 0.0272, 0.0256, 0.0255, 0.0237, 0.0227, 0.0222, 0.0213,
        0.0193, 0.0176, 0.0165, 0.0154, 0.0142, 0.0131, 0.0124, 0.0116, 0.0102,
        0.0096, 0.0095, 0.0089, 0.0082, 0.0078, 0.0071, 0.0064, 0.0062, 0.0061,
        0.0056, 0.0056, 0.0051, 0.0049, 0.0045, 0.0044, 0.0042, 0.0040, 0.0038,
        0.0037, 0.0035, 0.0035, 0.0034, 0.0032, 0.0030, 0.0029, 0.0028, 0.0027,
        0.0026, 0.0025, 0.0025, 0.0024, 0.0024, 0.0023, 0.0021, 0.0021, 0.0020,
        0.0020, 0.0020, 0.0019, 0.0018, 0.0017, 0.0017, 0.0017, 0.0016, 0.0016,
        0.0015, 0.0015, 0.0015, 0.0014, 0.0014, 0.0014, 0.0013, 0.0013, 0.0013,
        0.0012, 0.0012, 0.0012, 0.0012, 0.0011, 0.0011, 0.0011, 0.0011, 0.0010,
        0.0010, 0.0010, 0.0010, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0008,
        0.0008, 0.0008, 0.0008, 0.0008, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007,
        0.0007, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0005,
        0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0004, 0.0004, 0.0004, 0.0004,
        0.0004, 0.0004]), 'text_variances': torch.torch.tensor([0.0618, 0.0596, 0.0518, 0.0508, 0.0446, 0.0435, 0.0371, 0.0360, 0.0350,
        0.0325, 0.0299, 0.0270, 0.0268, 0.0253, 0.0239, 0.0235, 0.0230, 0.0215,
        0.0196, 0.0193, 0.0156, 0.0149, 0.0144, 0.0139, 0.0126, 0.0114, 0.0109,
        0.0103, 0.0097, 0.0091, 0.0083, 0.0078, 0.0077, 0.0070, 0.0065, 0.0063,
        0.0060, 0.0053, 0.0052, 0.0051, 0.0048, 0.0046, 0.0044, 0.0043, 0.0041,
        0.0039, 0.0035, 0.0033, 0.0033, 0.0031, 0.0029, 0.0029, 0.0028, 0.0027,
        0.0025, 0.0024, 0.0022, 0.0021, 0.0021, 0.0020, 0.0019, 0.0019, 0.0018,
        0.0017, 0.0017, 0.0016, 0.0015, 0.0015, 0.0015, 0.0014, 0.0013, 0.0013,
        0.0013, 0.0012, 0.0012, 0.0012, 0.0012, 0.0011, 0.0011, 0.0011, 0.0010,
        0.0010, 0.0010, 0.0009, 0.0009, 0.0009, 0.0009, 0.0008, 0.0008, 0.0008,
        0.0008, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0006, 0.0006,
        0.0006, 0.0006, 0.0006, 0.0006, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
        0.0005, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004,
        0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003, 0.0003,
        0.0002, 0.0002]), 'uniformity_loss': {'image_uniformity_loss': torch.torch.tensor(-3.6482, device='cpu'), 'text_uniformity_loss': torch.torch.tensor(-3.6334, device='cpu'), 'total_uniformity_loss': torch.torch.tensor(-3.6408, device='cpu'), 'cross_encoder_uniform_loss': torch.torch.tensor(-3.6791, device='cpu')}, 'alignment_loss': torch.torch.tensor(0.5359, device='cpu')}}

,
    {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'mean_cosine_similarity': 0.6534503102302551, 'linear_seperability_accuracy': 0.8011077542799597, 'centroid_euclidean_distance': 0.11362453550100327, 'val_image_classification_acc': {1: 0.2664248286981056, 3: 0.4615074566706973, 5: 0.5592503022974608, 10: 0.6920596533655784}, 'get_val_image_retrieval_acc': {1: 0.2750906892382104, 3: 0.4681580008061266, 5: 0.5628778718258767, 10: 0.6878274889157597}, 'image_variances': torch.torch.tensor([0.0374, 0.0364, 0.0338, 0.0312, 0.0301, 0.0282, 0.0276, 0.0265, 0.0257,
        0.0246, 0.0243, 0.0236, 0.0234, 0.0227, 0.0227, 0.0220, 0.0213, 0.0210,
        0.0206, 0.0193, 0.0183, 0.0180, 0.0176, 0.0172, 0.0169, 0.0159, 0.0145,
        0.0142, 0.0138, 0.0130, 0.0129, 0.0122, 0.0113, 0.0112, 0.0108, 0.0103,
        0.0098, 0.0095, 0.0089, 0.0085, 0.0080, 0.0074, 0.0074, 0.0069, 0.0066,
        0.0064, 0.0061, 0.0056, 0.0055, 0.0052, 0.0049, 0.0048, 0.0047, 0.0044,
        0.0044, 0.0041, 0.0040, 0.0039, 0.0037, 0.0036, 0.0035, 0.0034, 0.0032,
        0.0032, 0.0031, 0.0030, 0.0029, 0.0027, 0.0025, 0.0025, 0.0024, 0.0024,
        0.0024, 0.0022, 0.0022, 0.0022, 0.0021, 0.0021, 0.0020, 0.0019, 0.0019,
        0.0018, 0.0018, 0.0017, 0.0017, 0.0017, 0.0016, 0.0016, 0.0016, 0.0015,
        0.0015, 0.0015, 0.0014, 0.0014, 0.0014, 0.0013, 0.0013, 0.0012, 0.0012,
        0.0012, 0.0011, 0.0011, 0.0011, 0.0011, 0.0010, 0.0010, 0.0010, 0.0010,
        0.0010, 0.0009, 0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0008,
        0.0007, 0.0007, 0.0007, 0.0007, 0.0007, 0.0006, 0.0006, 0.0006, 0.0006,
        0.0005, 0.0005]), 'text_variances': torch.torch.tensor([0.0366, 0.0349, 0.0338, 0.0311, 0.0303, 0.0293, 0.0279, 0.0271, 0.0269,
        0.0256, 0.0251, 0.0236, 0.0233, 0.0226, 0.0225, 0.0219, 0.0217, 0.0212,
        0.0204, 0.0201, 0.0187, 0.0182, 0.0176, 0.0174, 0.0173, 0.0168, 0.0163,
        0.0159, 0.0147, 0.0141, 0.0132, 0.0129, 0.0119, 0.0117, 0.0115, 0.0106,
        0.0101, 0.0100, 0.0090, 0.0085, 0.0083, 0.0082, 0.0078, 0.0074, 0.0070,
        0.0064, 0.0064, 0.0062, 0.0057, 0.0056, 0.0051, 0.0050, 0.0046, 0.0045,
        0.0042, 0.0040, 0.0039, 0.0037, 0.0036, 0.0034, 0.0032, 0.0030, 0.0029,
        0.0029, 0.0028, 0.0026, 0.0025, 0.0025, 0.0024, 0.0023, 0.0022, 0.0021,
        0.0020, 0.0019, 0.0019, 0.0019, 0.0019, 0.0018, 0.0018, 0.0017, 0.0016,
        0.0015, 0.0015, 0.0015, 0.0015, 0.0014, 0.0014, 0.0013, 0.0012, 0.0012,
        0.0012, 0.0011, 0.0011, 0.0011, 0.0011, 0.0010, 0.0010, 0.0010, 0.0009,
        0.0009, 0.0009, 0.0008, 0.0008, 0.0008, 0.0008, 0.0007, 0.0007, 0.0007,
        0.0007, 0.0007, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0006, 0.0005,
        0.0005, 0.0005, 0.0005, 0.0005, 0.0004, 0.0004, 0.0004, 0.0004, 0.0004,
        0.0003, 0.0003]), 'uniformity_loss': {'image_uniformity_loss': torch.torch.tensor(-3.7702, device='cpu'), 'text_uniformity_loss': torch.torch.tensor(-3.7536, device='cpu'), 'total_uniformity_loss': torch.torch.tensor(-3.7619, device='cpu'), 'cross_encoder_uniform_loss': torch.torch.tensor(-3.8161, device='cpu')}, 'alignment_loss': torch.torch.tensor(0.6931, device='cpu')}}



]

zs_datas = [
    {'checkpoint_path': 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'imagenet_zs_acc': 0.1205, 'dtd_zs_acc': 0.09787234042553192, 'caltech101_zs_acc': 0.38285121585801546, 'cifar10_zs_acc': 0.7515, 'cifar100_zs_acc': 0.3039},
     'uniformity_loss': 0,
     'clip_projection_dim': 32},

     {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt', 'gap_stuff': {'imagenet_zs_acc': 0.12558, 'dtd_zs_acc': 0.07872340425531915, 'caltech101_zs_acc': 0.4292958395758903, 'cifar10_zs_acc': 0.7396, 'cifar100_zs_acc': 0.2991},
     'uniformity_loss': 1,
     'clip_projection_dim': 32},

     {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'imagenet_zs_acc': 0.1345, 'dtd_zs_acc': 0.10691489361702128, 'caltech101_zs_acc': 0.47055433905727784, 'cifar10_zs_acc': 0.7417, 'cifar100_zs_acc': 0.3247},
      'uniformity_loss': 2,
     'clip_projection_dim': 32},

     {'checkpoint_path': 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'imagenet_zs_acc': 0.1374, 'dtd_zs_acc': 0.08776595744680851, 'caltech101_zs_acc': 0.3586493027544082, 'cifar10_zs_acc': 0.7547, 'cifar100_zs_acc': 0.3274},
      'uniformity_loss': 0,
     'clip_projection_dim': 64},

     {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt', 'gap_stuff': {'imagenet_zs_acc': 0.13806, 'dtd_zs_acc': 0.08191489361702127, 'caltech101_zs_acc': 0.4568399216319004, 'cifar10_zs_acc': 0.777, 'cifar100_zs_acc': 0.33},
      'uniformity_loss': 1,
     'clip_projection_dim': 64},

     {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'imagenet_zs_acc': 0.15074, 'dtd_zs_acc': 0.09414893617021276, 'caltech101_zs_acc': 0.44877261726403134, 'cifar10_zs_acc': 0.7727, 'cifar100_zs_acc': 0.3471},
      'uniformity_loss': 2,
     'clip_projection_dim': 64},

     {'checkpoint_path': 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'imagenet_zs_acc': 0.1358, 'imagenet_gap_stuff': {'image_uniformity_loss': torch.tensor(-1.7893, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.4251, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.6800, device='cpu'), 'inter_modality_loss': torch.tensor(6.7873, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(5.4919, device='cpu')}, 'dtd_zs_acc': 0.10851063829787234, 'dtd_gap_stuff': {'image_uniformity_loss': torch.tensor(-1.2020, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.5544, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.7281, device='cpu'), 'inter_modality_loss': torch.tensor(3.8350, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(4.1123, device='cpu')}, 'caltech101_zs_acc': 0.41558142215051286, 'caltech101_gap_stuff': {'image_uniformity_loss': torch.tensor(-1.6440, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.4523, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.7000, device='cpu'), 'inter_modality_loss': torch.tensor(4.5273, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(2.6520, device='cpu')}, 'cifar10_zs_acc': 0.732, 'cifar10_gap_stuff': {'image_uniformity_loss': torch.tensor(-1.2072, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.4953, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.7400, device='cpu'), 'inter_modality_loss': torch.tensor(2.2052, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(0.9280, device='cpu')}, 'cifar100_zs_acc': 0.3289, 'cifar100_gap_stuff': {'image_uniformity_loss': torch.tensor(-1.2190, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.4949, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.7105, device='cpu'), 'inter_modality_loss': torch.tensor(4.5125, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(3.5497, device='cpu')}},
      'uniformity_loss': 0,
     'clip_projection_dim': 128},

     {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt', 'gap_stuff': {'imagenet_zs_acc': 0.14016, 'imagenet_gap_stuff': {'image_uniformity_loss': torch.tensor(-3.2279, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.1534, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.5102, device='cpu'), 'inter_modality_loss': torch.tensor(6.5585, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(12.1235, device='cpu')}, 'dtd_zs_acc': 0.11436170212765957, 'dtd_gap_stuff': {'image_uniformity_loss': torch.tensor(-1.9927, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.4663, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.6165, device='cpu'), 'inter_modality_loss': torch.tensor(3.8105, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(8.0155, device='cpu')}, 'caltech101_zs_acc': 0.485421228535208, 'caltech101_gap_stuff': {'image_uniformity_loss': torch.tensor(-2.8929, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.1866, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.5769, device='cpu'), 'inter_modality_loss': torch.tensor(4.3091, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(5.0157, device='cpu')}, 'cifar10_zs_acc': 0.7678, 'cifar10_gap_stuff': {'image_uniformity_loss': torch.tensor(-2.4390, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.2553, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.4705, device='cpu'), 'inter_modality_loss': torch.tensor(1.9535, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(2.4386, device='cpu')}, 'cifar100_zs_acc': 0.3418, 'cifar100_gap_stuff': {'image_uniformity_loss': torch.tensor(-2.3994, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.2543, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.4840, device='cpu'), 'inter_modality_loss': torch.tensor(4.3106, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(9.0317, device='cpu')}},
      'uniformity_loss': 1,
     'clip_projection_dim': 128},

     {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'imagenet_zs_acc': 0.15466, 'imagenet_gap_stuff': {'image_uniformity_loss': torch.tensor(-3.4139, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.1104, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.5068, device='cpu'), 'inter_modality_loss': torch.tensor(6.5866, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(11.9859, device='cpu')}, 'dtd_zs_acc': 0.08670212765957447, 'dtd_gap_stuff': {'image_uniformity_loss': torch.tensor(-2.1259, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.3582, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.7397, device='cpu'), 'inter_modality_loss': torch.tensor(3.8147, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(9.0661, device='cpu')}, 'caltech101_zs_acc': 0.5055894894548807, 'caltech101_gap_stuff': {'image_uniformity_loss': torch.tensor(-3.0493, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.1348, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.6281, device='cpu'), 'inter_modality_loss': torch.tensor(4.3300, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(4.6833, device='cpu')}, 'cifar10_zs_acc': 0.7461, 'cifar10_gap_stuff': {'image_uniformity_loss': torch.tensor(-2.5984, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.2046, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.5222, device='cpu'), 'inter_modality_loss': torch.tensor(1.9769, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(3.2257, device='cpu')}, 'cifar100_zs_acc': 0.3607, 'cifar100_gap_stuff': {'image_uniformity_loss': torch.tensor(-2.5751, device='cpu'), 'mean_cosine_similarity': torch.tensor(0.1940, device='cpu'), 'centroid_euclidean_distance': torch.tensor(0.5610, device='cpu'), 'inter_modality_loss': torch.tensor(4.3343, device='cpu'), 'temp_scaled_inter_modality_loss': torch.tensor(8.6442, device='cpu')}},
      'uniformity_loss': 2,
     'clip_projection_dim': 128}










]



lp_datas = [

    {'checkpoint_path': 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'dtd_lp_acc': 0.624468085106383, 'caltech101_lp_acc': 0.9184763719906651, 'cifar10_lp_acc': 0.9082, 'cifar100_lp_acc': 0.678},
    'uniformity_loss': 0,
    'clip_projection_dim': 128},

    {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt', 'gap_stuff': {'dtd_lp_acc': 0.6319148936170212, 'caltech101_lp_acc': 0.9180665000972588, 'cifar10_lp_acc': 0.9108, 'cifar100_lp_acc': 0.6806},
    'uniformity_loss': 1,
    'clip_projection_dim': 128},

    {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_128_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'dtd_lp_acc': 0.6351063829787233, 'caltech101_lp_acc': 0.9128620851605047, 'cifar10_lp_acc': 0.912, 'cifar100_lp_acc': 0.6742},
    'uniformity_loss': 2,
    'clip_projection_dim': 128},

    {'checkpoint_path': 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'dtd_lp_acc': 0.6319148936170212, 'caltech101_lp_acc': 0.9131892611713329, 'cifar10_lp_acc': 0.904, 'cifar100_lp_acc': 0.654},
    'uniformity_loss': 0,
    'clip_projection_dim': 64},


    {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt', 'gap_stuff': {'dtd_lp_acc': 0.6191489361702127, 'caltech101_lp_acc': 0.9196414772785833, 'cifar10_lp_acc': 0.9028, 'cifar100_lp_acc': 0.659},
    'uniformity_loss': 1,
    'clip_projection_dim': 64},


    {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_64_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'dtd_lp_acc': 0.6542553191489362, 'caltech101_lp_acc': 0.9200856527368112, 'cifar10_lp_acc': 0.9108, 'cifar100_lp_acc': 0.6576},
    'uniformity_loss': 2,
    'clip_projection_dim': 64},

    {'checkpoint_path': 'checkpoints/T0.01_Lit_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'dtd_lp_acc': 0.6351063829787233, 'caltech101_lp_acc': 0.9139760323512107, 'cifar10_lp_acc': 0.907, 'cifar100_lp_acc': 0.6842},
    'uniformity_loss': 0,
    'clip_projection_dim': 32},

    {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_FINAL3.pt', 'gap_stuff': {'dtd_lp_acc': 0.65, 'caltech101_lp_acc': 0.9174563544326593, 'cifar10_lp_acc': 0.9152, 'cifar100_lp_acc': 0.6806},
    'uniformity_loss': 1,
    'clip_projection_dim': 32},

    {'checkpoint_path': 'checkpoints/T0.01_Lituniform_align_xuniform_42_finetune_I1C2E1E2_32_val_as_val_512_mscoco_VIT_pretrained_EVAL.pt', 'gap_stuff': {'dtd_lp_acc': 0.6404255319148936, 'caltech101_lp_acc': 0.9180491211719559, 'cifar10_lp_acc': 0.9032, 'cifar100_lp_acc': 0.6776},
    'uniformity_loss': 2,
    'clip_projection_dim': 32}















]


plt.rcParams.update({'font.size': 18})

legend_font_size = "22"


def plot_explained_vars(summaries):
      

  width=0.45


  all_average_variances = []


  for i, summary in enumerate(summaries):
        

        
        avg_vars = []

        for j in range(8):
            
            average_variance = (summary[f'image_variance{j}'] + summary[f'text_variance{j}']) / 2
            avg_vars.append(average_variance)

        all_average_variances.append(avg_vars)

  for i, summary in enumerate(summaries):
      
      # name = summary['name']
      name = names[i]
      color = colors[i]
      plt.bar(np.arange(8) + i * width, all_average_variances[i], width, label=r'${}$'.format(name), color=f'{color}')

    
      
  plt.xticks(np.arange(8) + width / 2, ('0-7', '8-15', '16-23', '24-31', '32-39', '40-47', '48-55', '56-63'))

  plt.legend(fontsize="22")

  plt.xlabel('Dimensions')

  plt.ylabel('PCA Explained variance')



  plt.show()


def plot_cumulative_explained_vars():

    
  
  
    all_average_variances = []
  
  
    for i, data in enumerate(datas):
          
          avg_vars = [0]
  
          for j in range(len(data['gap_stuff'][f'image_variances'])):
              
              average_variance = (data['gap_stuff'][f'image_variances'][j] + data['gap_stuff'][f'text_variances'][j]) / 2
              avg_vars.append(average_variance)

          for j in range(128-len(avg_vars)):
              avg_vars.append(0)
              
  
          all_average_variances.append(avg_vars)
  
    for i, run in enumerate(datas):
        
        # name = summary['name']
        name = names[i]
        color = colors[i]
        # plt.plot(np.arange(start=1, stop=64, step=8), np.cumsum(all_average_variances[i]), label=r'${}$'.format(name), color=f'{color}')
        plt.plot(np.arange(start=0, stop=129, step=1), np.cumsum(all_average_variances[i]), label=r'${}$'.format(name), color=f'{color}')
  
      
        
    # plt.xticks(np.arange(8) + width / 2, ('0-7', '8-15', '16-23', '24-31', '32-39', '40-47', '48-55', '56-63'))
  
    plt.legend(fontsize="22")
  
    plt.xlabel('Dimensions')
  
    # plt.ylabel('PCA Explained variance')

    # set title
    plt.title('Cumulative PCA Explained Variance')

    # add grid 
    plt.grid()



  
    plt.show()





def plot_dim_acc_graph(runs, fields=['dtd_zs_acc', 'caltech101_zs_acc', 'cifar10_zs_acc', 'cifar100_zs_acc'], group_by='uniformity_loss', xlabel='CLIP Dimensionality', ylabel='CIFAR10 Validation Accuracy', group_labels=['L_{CLIP}', 'L_{CLIP+Align+Uniform+XUniform}'], markers=['o', 'x'], colors=['tab:blue', 'tab:red']):
    

    collected_metric_values = []

    collected_dims = []

    collected_groups = []

    
    # gather data
    for i, run in enumerate(runs):
        
        collected_dims.append(run['clip_projection_dim'])
        # collected_metric_values.append(run['summary'][field])
        collected_groups.append(run[group_by])

    # count unique group byes
    unique_groups = list(set(collected_groups))

    # count unique dimensionalities
    dimensionalities = list(set(collected_dims))

    # for each unique group by, get all metrics, sorted by dimensionalities
    for group in unique_groups:
        
        all_metrics = []

        for dim in dimensionalities:
            
            metrics = []

            for i, run in enumerate(runs):
                
                if run['clip_projection_dim'] == dim and run[group_by] == group:
                    
                    avg_metric = 0
                    
                    for field in fields:
                      avg_metric += run['gap_stuff'][field]

                    avg_metric /= len(fields)
                    metrics.append(avg_metric)

            all_metrics.append(metrics)



        # sort by dimensionalities
        all_metrics = [x for _, x in sorted(zip(dimensionalities, all_metrics))]
        dimensionalities = sorted(dimensionalities)

        plt.plot(dimensionalities, all_metrics, marker= markers[unique_groups.index(group)], label=r'${}$'.format(group_labels[unique_groups.index(group)]), color=colors[unique_groups.index(group)], markersize=10)
        # plt.plot(dimensionalities, all_metrics, label=r'${}$'.format(group), marker= markers[unique_groups.index(group)], color=colors[unique_groups.index(group)])
        
    plt.legend(fontsize=legend_font_size)
    plt.xlabel(xlabel)
    # plt.ylabel(ylabel)

    plt.title(f'{ylabel}')
    # draw grid
    plt.grid()
    plt.show()


    
    pass



# plot_explained_vars(summaries)

# print(runs[-2:])

# plot_cumulative_explained_vars()

plot_dim_acc_graph(zs_datas, fields=['dtd_zs_acc', 'caltech101_zs_acc', 'cifar10_zs_acc', 'cifar100_zs_acc', 'imagenet_zs_acc'], group_by='uniformity_loss', xlabel='CLIP Dimensionality', ylabel='Average Zero-Shot Accuracy', group_labels=names, markers=['o', 'x', 'x'], colors=['tab:blue', 'tab:red', 'tab:green'])
# plot_dim_acc_graph(lp_datas, fields=['dtd_lp_acc', 'caltech101_lp_acc', 'cifar10_lp_acc', 'cifar100_lp_acc'], group_by='uniformity_loss', xlabel='CLIP Dimensionality', ylabel='Average Linear Probe Accuracy', group_labels=names, markers=['o', 'x', 'x'], colors=['tab:blue', 'tab:red', 'tab:green'])


# old
# plot_dim_acc_graph(runs, fields=['dtd_zs_acc', 'caltech101_zs_acc', 'cifar10_zs_acc', 'cifar100_zs_acc'], group_by='uniformity_loss', xlabel='CLIP Dimensionality', ylabel='Average Linear Probe Accuracy', group_labels=names, markers=['o', 'x', 'x'], colors=['tab:blue', 'tab:red'])
# plot_dim_acc_graph(runs, field='val_image_classification_accuracy', group_by='uniformity_loss', xlabel='CLIP Dimensionality', ylabel='I -> T', group_labels=['L_{CLIP}', 'L_{CLIP+Align+Uniform}'], markers=['o', 'x'], colors=['tab:blue', 'tab:red'])