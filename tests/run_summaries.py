runs = [


    { # 3D CLIP
        'summary': {
            "cifar10_mean_cosine_similarity": 0.6531944274902344,
            "non_similar_mean_cosine_similarity": 0.11855721473693848,
            "image_S7": 0,
            "image_variance1": 0.37559619545936584,
            "train_pearson_loss": -100,
            "train_cyclic_dir_loss": 0.04213554412126541,
            "train_cross_uniformity_loss": -1.8288499116897583,
            "average_linear_probe_accuracy": -1,
            "svd": -100,
            "text_rank": 3,
            "text_variance4": 0,
            "text_variance6": 0,
            "train_intramodality_loss": -100,
            "train_alignment_loss": 0.06374652683734894,
            "rsa_before_interchanging": 0.9328194924107048,
            "image_rank": 3,
            "text_variance2": 0.2003524154424667,
            "image_variance2": 0.20258593559265137,
            "image_variance3": 0,
            "image_variance5": 0,
            "image_variance7": 0,
            "centroid_euclidean_distance": 0.023306915536522865,
            "pearson_image_intermodality_rsa": 0.9656607058804146,
            "mean_pairwise_euclidean_distance": 0.17363889515399933,
            "_runtime": 8142.595180988312,
            "uniformity": -1.8284034729003904,
            "temperature": 0.010000000000000004,
            "train_uniformity_loss": -1.8284034729003904,
            "val_image_retrieval_accuracy": 0.20703125,
            "text_S7": 0,
            "image_S4": 0,
            "_timestamp": 1716178776.938702,
            "val_image_classification_accuracy": 0.201171875,
            "_wandb.runtime": 8181,
            "image_S6": 0,
            "image_variance0": 0.4218178987503052,
            "text_intermodality_rsa": 0.968104191897271,
            "dtd_linear_probe_accuracy": -1,
            "std_dev_linear_probe_accuracy": 0,
            "text_S4": 0,
            "text_S6": 0,
            "image_S5": 0,
            "train_total_loss": 4.624942779541016,
            "pearson_text_intermodality_rsa": 0.9694824465255034,
            "avg_S": 13.028800964355469,
            "text_S1": 13.495500564575195,
            "image_S2": 11.959301948547363,
            "image_S3": 0,
            "full_text_rank": 3,
            "first_lt1_value": -1,
            "_step": 11000,
            "text_S5": 0,
            "cifar10_val_image_classification_accuracy": 0.5689,
            "cifar100_val_image_classification_accuracy": 0.0923,
            "image_variance6": 0,
            "mean_text_text_cosine_similarity": 0.11106877028942108,
            "pearson_rsa_before_interchanging": 0.9361472083802668,
            "average_intra_modality_cosine_similarity": 0.11876698583364488,
            "cifar10_image_uniformity_loss": -0.8699503540992737,
            "text_S0": 13.89645767211914,
            "text_S2": 11.694446563720703,
            "text_variance1": 0.3745771050453186,
            "image_variance4": 0,
            "image_intermodality_rsa": 0.9634153144878334,
            "train_uniform_cyclic_loss": 0.1663145124912262,
            "image_S1": 13.438546180725098,
            "text_variance5": 0,
            "train_rsa_loss": -100,
            "linear_seperability_accuracy": 0.5,
            "mean_image_image_cosine_similarity": 0.12646520137786865,
            "cifar10_centroid_euclidean_distance": 0.14606738090515137,
            "text_variance3": 0,
            "train_cyclic_loss": 10.58649765625001,
            "cifar100_linear_probe_accuracy": 0.6782,
            "text_variance0": 0.4250704348087311,
            "text_variance7": 0,
            "full_image_rank": 3,
            "centroid_cosine_similarity": 0.999780535697937,
            "cifar10_linear_probe_accuracy": 0.9034,
            "cifar10_temp_scaled_inter_modality_loss": 6.804633140563965,
            "text_S3": 0,
            "image_S0": 13.72518539428711,
            "caltech101_linear_probe_accuracy": -1,
            "mean_cosine_similarity": 0.9681267142295836,
            "train_intermodality_loss": 4.624942779541016,
            "cifar10_inter_modality_loss": 2.0836191177368164
            },
        'config': {
        "lr": {
            "desc": None,
            "value": 0.00001
        },
        "host": {
            "desc": None,
            "value": "cirrus"
        },
        "seed": {
            "desc": None,
            "value": 42
        },
        "_wandb": {
            "desc": None,
            "value": {
            "t": {
                "1": [
                1,
                5,
                11,
                41,
                49,
                51,
                53,
                55,
                100
                ],
                "2": [
                1,
                5,
                11,
                41,
                49,
                51,
                53,
                55,
                100
                ],
                "3": [
                2,
                23,
                37
                ],
                "4": "3.10.13",
                "5": "0.16.0",
                "6": "4.34.0",
                "8": [
                5
                ],
                "13": "linux-x86_64"
            },
            "framework": "huggingface",
            "start_time": 1716170634.343521,
            "cli_version": "0.16.0",
            "is_jupyter_run": False,
            "python_version": "3.10.13",
            "is_kaggle_kernel": False,
            "huggingface_version": "4.34.0"
            }
        },
        "dataset": {
            "desc": None,
            "value": "mscoco"
        },
        "csv_path": {
            "desc": None,
            "value": "stats/"
        },
        "n_epochs": {
            "desc": None,
            "value": 6
        },
        "rsa_loss": {
            "desc": None,
            "value": False
        },
        "svd_loss": {
            "desc": None,
            "value": False
        },
        "max_steps": {
            "desc": None,
            "value": None
        },
        "batch_size": {
            "desc": None,
            "value": 64
        },
        "grad_cache": {
            "desc": None,
            "value": False
        },
        "save_every": {
            "desc": None,
            "value": 200
        },
        "W_layer_gap": {
            "desc": None,
            "value": -1
        },
        "cifar10_acc": {
            "desc": None,
            "value": True
        },
        "cuda_device": {
            "desc": None,
            "value": "cuda:0"
        },
        "cyclip_loss": {
            "desc": None,
            "value": False
        },
        "num_workers": {
            "desc": None,
            "value": 12
        },
        "one_encoder": {
            "desc": None,
            "value": False
        },
        "same_inputs": {
            "desc": None,
            "value": False
        },
        "save_losses": {
            "desc": None,
            "value": False
        },
        "simclr_loss": {
            "desc": None,
            "value": False
        },
        "temperature": {
            "desc": None,
            "value": 0.01
        },
        "loss_weights": {
            "desc": None,
            "value": {
            "image_to_text_weight": 0.5,
            "text_to_image_weight": 0.5
            }
        },
        "pearson_loss": {
            "desc": None,
            "value": False
        },
        "same_encoder": {
            "desc": None,
            "value": False
        },
        "vision_model": {
            "desc": None,
            "value": "VIT"
        },
        "weight_decay": {
            "desc": None,
            "value": 0.1
        },
        "hf_clip_model": {
            "desc": None,
            "value": "openai/clip-vit-base-patch32"
        },
        "use_scheduler": {
            "desc": None,
            "value": "EXP"
        },
        "alignment_loss": {
            "desc": None,
            "value": False
        },
        "n_warmup_steps": {
            "desc": None,
            "value": 10000
        },
        "schedule_every": {
            "desc": None,
            "value": 400
        },
        "uniformity_loss": {
            "desc": None,
            "value": False
        },
        "cifar_batch_size": {
            "desc": None,
            "value": 128
        },
        "do_checkpointing": {
            "desc": None,
            "value": True
        },
        "mismatched_pairs": {
            "desc": None,
            "value": False
        },
        "n_embeds_to_save": {
            "desc": None,
            "value": 512
        },
        "use_train_as_val": {
            "desc": None,
            "value": False
        },
        "cosine_align_loss": {
            "desc": None,
            "value": False
        },
        "encoder1_modality": {
            "desc": None,
            "value": "image"
        },
        "encoder2_modality": {
            "desc": None,
            "value": "text"
        },
        "openai_clip_model": {
            "desc": None,
            "value": "ViT-B/32"
        },
        "scaled_denominator": {
            "desc": None,
            "value": False
        },
        "train_from_scratch": {
            "desc": None,
            "value": False
        },
        "clip_projection_dim": {
            "desc": None,
            "value": 3
        },
        "intra_modality_loss": {
            "desc": None,
            "value": False
        },
        "selected_clip_model": {
            "desc": None,
            "value": "clip_finetuned_temp"
        },
        "uniform_cyclic_loss": {
            "desc": None,
            "value": False
        },
        "train_only_one_batch": {
            "desc": None,
            "value": False
        },
        "use_cached_val_batch": {
            "desc": None,
            "value": True
        },
        "visualize_embeddings": {
            "desc": None,
            "value": False
        },
        "cross_uniformity_loss": {
            "desc": None,
            "value": False
        },
        "cyclic_direction_loss": {
            "desc": None,
            "value": False
        },
        "grad_cache_multiplier": {
            "desc": None,
            "value": 16
        },
        "learnable_temperature": {
            "desc": None,
            "value": False
        },
        "second_caption_offset": {
            "desc": None,
            "value": False
        },
        "show_incorrect_images": {
            "desc": None,
            "value": False
        },
        "train_from_pretrained": {
            "desc": None,
            "value": True
        },
        "use_small_trainloader": {
            "desc": None,
            "value": False
        },
        "validation_batch_size": {
            "desc": None,
            "value": 512
        },
        "cosine_uniformity_loss": {
            "desc": None,
            "value": False
        },
        "delete_val_batch_first": {
            "desc": None,
            "value": False
        },
        "finetune_clip_backbone": {
            "desc": None,
            "value": True
        },
        "common_projection_layer": {
            "desc": None,
            "value": False
        },
        "loss_file_name_template": {
            "desc": None,
            "value": "Ttemp_loss_seed_trainmode_captionencoder_dim_val_bsize_dataset_vmodel_pretrained_FINAL"
        },
        "remove_contrastive_loss": {
            "desc": None,
            "value": False
        },
        "validation_dataset_size": {
            "desc": None,
            "value": 512
        },
        "continue_from_checkpoint": {
            "desc": None,
            "value": False
        },
        "shared_transformer_layers": {
            "desc": None,
            "value": False
        },
        "zero_shot_acc_num_workers": {
            "desc": None,
            "value": 4
        },
        "intra_modality_temperature": {
            "desc": None,
            "value": 0.01
        },
        "save_encoder_hidden_states": {
            "desc": None,
            "value": False
        },
        "finetune_multi_layer_projection": {
            "desc": None,
            "value": False
        }
    }
    },
    {
        'summary': {
            "train_cyclic_loss": 14.367763281250014,
            "cifar10_image_uniformity_loss": -1.2098475694656372,
            "val_image_classification_accuracy": 0.189453125,
            "image_S4": 0,
            "train_rsa_loss": -100,
            "pearson_rsa_before_interchanging": 0.916121216011962,
            "text_S5": 0,
            "image_S7": 0,
            "first_lt1_value": -1,
            "dtd_linear_probe_accuracy": -1,
            "image_rank": 3,
            "non_similar_mean_cosine_similarity": 0.008592190220952034,
            "average_intra_modality_cosine_similarity": 0.008978351950645447,
            "temperature": 0.010000000000000004,
            "full_text_rank": 3,
            "text_variance7": 0,
            "image_variance6": 0,
            "text_variance2": 0.2969129681587219,
            "train_pearson_loss": -100,
            "image_intermodality_rsa": 0.9547311881617894,
            "average_linear_probe_accuracy": -1,
            "_step": 11000,
            "avg_S": 13.051661491394045,
            "text_S3": 0,
            "_timestamp": 1716180549.0008576,
            "mean_text_text_cosine_similarity": 0.006239277310669422,
            "text_intermodality_rsa": 0.9592636638793032,
            "centroid_euclidean_distance": 0.030676987022161484,
            "std_dev_linear_probe_accuracy": 0,
            "text_S0": 13.633450508117676,
            "text_variance5": 0,
            "image_variance7": 0,
            "train_uniformity_loss": -2.053398609161377,
            "val_image_retrieval_accuracy": 0.18359375,
            "cifar10_val_image_classification_accuracy": 0.5673,
            "cifar100_val_image_classification_accuracy": 0.0911,
            "uniformity": -2.053398609161377,
            "image_variance1": 0.3437744379043579,
            "image_variance4": 0,
            "cifar10_inter_modality_loss": 1.943750500679016,
            "image_S3": 0,
            "image_variance0": 0.35954874753952026,
            "train_total_loss": 3.609121322631836,
            "mean_cosine_similarity": 0.9564130902290344,
            "caltech101_linear_probe_accuracy": -1,
            "image_S6": 0,
            "text_variance0": 0.3612006902694702,
            "train_alignment_loss": 0.08717383444309235,
            "train_uniform_cyclic_loss": 0.22449304163455963,
            "image_variance5": 0,
            "rsa_before_interchanging": 0.9160279396380168,
            "centroid_cosine_similarity": 0.9883947968482972,
            "pearson_text_intermodality_rsa": 0.9592926600474382,
            "text_S6": 0,
            "text_variance1": 0.34188637137413025,
            "full_image_rank": 3,
            "image_variance3": 0,
            "cifar10_centroid_euclidean_distance": 0.2026257961988449,
            "cifar10_temp_scaled_inter_modality_loss": 8.807262420654297,
            "mean_pairwise_euclidean_distance": 0.20041972398757937,
            "mean_image_image_cosine_similarity": 0.011717426590621471,
            "_wandb.runtime": 9968,
            "image_S0": 13.592368125915527,
            "image_S5": 0,
            "image_variance2": 0.29667675495147705,
            "cifar10_linear_probe_accuracy": 0.8992,
            "pearson_image_intermodality_rsa": 0.9548301517471268,
            "image_S1": 13.284838676452637,
            "image_S2": 12.278457641601562,
            "train_cyclic_dir_loss": 0.05748851224780083,
            "train_intramodality_loss": -100,
            "text_variance6": 0,
            "train_intermodality_loss": 5.575345993041992,
            "cifar10_mean_cosine_similarity": 0.4232771694660187,
            "text_S1": 13.2377290725708,
            "text_S2": 12.283804893493652,
            "text_S4": 0,
            "text_variance4": 0,
            "svd": -100,
            "text_S7": 0,
            "_runtime": 9917.29094862938,
            "linear_seperability_accuracy": 0.5085365853658537,
            "text_rank": 3,
            "text_variance3": 0,
            "train_cross_uniformity_loss": -2.053682804107666,
            "cifar100_linear_probe_accuracy": 0.6678
            },
        'config': {
  "lr": {
    "desc": None,
    "value": 0.00001
  },
  "host": {
    "desc": None,
    "value": "cirrus"
  },
  "seed": {
    "desc": None,
    "value": 42
  },
  "_wandb": {
    "desc": None,
    "value": {
      "t": {
        "1": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          55,
          100
        ],
        "2": [
          1,
          5,
          11,
          41,
          49,
          51,
          53,
          55,
          100
        ],
        "3": [
          2,
          23,
          37
        ],
        "4": "3.10.13",
        "5": "0.16.0",
        "6": "4.34.0",
        "8": [
          5
        ],
        "13": "linux-x86_64"
      },
      "framework": "huggingface",
      "start_time": 1716170631.709909,
      "cli_version": "0.16.0",
      "is_jupyter_run": False,
      "python_version": "3.10.13",
      "is_kaggle_kernel": False,
      "huggingface_version": "4.34.0"
    }
  },
  "dataset": {
    "desc": None,
    "value": "mscoco"
  },
  "csv_path": {
    "desc": None,
    "value": "stats/"
  },
  "n_epochs": {
    "desc": None,
    "value": 6
  },
  "rsa_loss": {
    "desc": None,
    "value": False
  },
  "svd_loss": {
    "desc": None,
    "value": False
  },
  "max_steps": {
    "desc": None,
    "value": None
  },
  "batch_size": {
    "desc": None,
    "value": 64
  },
  "grad_cache": {
    "desc": None,
    "value": False
  },
  "save_every": {
    "desc": None,
    "value": 200
  },
  "W_layer_gap": {
    "desc": None,
    "value": -1
  },
  "cifar10_acc": {
    "desc": None,
    "value": True
  },
  "cuda_device": {
    "desc": None,
    "value": "cuda:1"
  },
  "cyclip_loss": {
    "desc": None,
    "value": False
  },
  "num_workers": {
    "desc": None,
    "value": 12
  },
  "one_encoder": {
    "desc": None,
    "value": False
  },
  "same_inputs": {
    "desc": None,
    "value": False
  },
  "save_losses": {
    "desc": None,
    "value": False
  },
  "simclr_loss": {
    "desc": None,
    "value": False
  },
  "temperature": {
    "desc": None,
    "value": 0.01
  },
  "loss_weights": {
    "desc": None,
    "value": {
      "image_to_text_weight": 0.5,
      "text_to_image_weight": 0.5
    }
  },
  "pearson_loss": {
    "desc": None,
    "value": False
  },
  "same_encoder": {
    "desc": None,
    "value": False
  },
  "vision_model": {
    "desc": None,
    "value": "VIT"
  },
  "weight_decay": {
    "desc": None,
    "value": 0.1
  },
  "hf_clip_model": {
    "desc": None,
    "value": "openai/clip-vit-base-patch32"
  },
  "use_scheduler": {
    "desc": None,
    "value": "EXP"
  },
  "alignment_loss": {
    "desc": None,
    "value": True
  },
  "n_warmup_steps": {
    "desc": None,
    "value": 10000
  },
  "schedule_every": {
    "desc": None,
    "value": 400
  },
  "uniformity_loss": {
    "desc": None,
    "value": True
  },
  "cifar_batch_size": {
    "desc": None,
    "value": 128
  },
  "do_checkpointing": {
    "desc": None,
    "value": True
  },
  "mismatched_pairs": {
    "desc": None,
    "value": False
  },
  "n_embeds_to_save": {
    "desc": None,
    "value": 512
  },
  "use_train_as_val": {
    "desc": None,
    "value": False
  },
  "cosine_align_loss": {
    "desc": None,
    "value": False
  },
  "encoder1_modality": {
    "desc": None,
    "value": "image"
  },
  "encoder2_modality": {
    "desc": None,
    "value": "text"
  },
  "openai_clip_model": {
    "desc": None,
    "value": "ViT-B/32"
  },
  "scaled_denominator": {
    "desc": None,
    "value": False
  },
  "train_from_scratch": {
    "desc": None,
    "value": False
  },
  "clip_projection_dim": {
    "desc": None,
    "value": 3
  },
  "intra_modality_loss": {
    "desc": None,
    "value": False
  },
  "selected_clip_model": {
    "desc": None,
    "value": "clip_finetuned_temp"
  },
  "uniform_cyclic_loss": {
    "desc": None,
    "value": False
  },
  "train_only_one_batch": {
    "desc": None,
    "value": False
  },
  "use_cached_val_batch": {
    "desc": None,
    "value": True
  },
  "visualize_embeddings": {
    "desc": None,
    "value": False
  },
  "cross_uniformity_loss": {
    "desc": None,
    "value": False
  },
  "cyclic_direction_loss": {
    "desc": None,
    "value": False
  },
  "grad_cache_multiplier": {
    "desc": None,
    "value": 16
  },
  "learnable_temperature": {
    "desc": None,
    "value": False
  },
  "second_caption_offset": {
    "desc": None,
    "value": False
  },
  "show_incorrect_images": {
    "desc": None,
    "value": False
  },
  "train_from_pretrained": {
    "desc": None,
    "value": True
  },
  "use_small_trainloader": {
    "desc": None,
    "value": False
  },
  "validation_batch_size": {
    "desc": None,
    "value": 512
  },
  "cosine_uniformity_loss": {
    "desc": None,
    "value": False
  },
  "delete_val_batch_first": {
    "desc": None,
    "value": False
  },
  "finetune_clip_backbone": {
    "desc": None,
    "value": True
  },
  "common_projection_layer": {
    "desc": None,
    "value": False
  },
  "loss_file_name_template": {
    "desc": None,
    "value": "Ttemp_loss_seed_trainmode_captionencoder_dim_val_bsize_dataset_vmodel_pretrained_FINAL"
  },
  "remove_contrastive_loss": {
    "desc": None,
    "value": False
  },
  "validation_dataset_size": {
    "desc": None,
    "value": 512
  },
  "continue_from_checkpoint": {
    "desc": None,
    "value": False
  },
  "shared_transformer_layers": {
    "desc": None,
    "value": False
  },
  "zero_shot_acc_num_workers": {
    "desc": None,
    "value": 4
  },
  "intra_modality_temperature": {
    "desc": None,
    "value": 0.01
  },
  "save_encoder_hidden_states": {
    "desc": None,
    "value": False
  },
  "finetune_multi_layer_projection": {
    "desc": None,
    "value": False
  }
}
    },

    { # 16D CLIP
        'summary': {
            "non_similar_mean_cosine_similarity": 0.5658016800880432,
            "text_variance2": 0.1519949585199356,
            "image_variance0": 0.26847487688064575,
            "image_variance1": 0.19117070734500885,
            "cifar100_linear_probe_accuracy": 0.6724,
            "temperature": 0.010000000000000004,
            "train_uniformity_loss": -1.580543041229248,
            "pearson_rsa_before_interchanging": 0.771684414902413,
            "image_variance7": 0.034809961915016174,
            "centroid_euclidean_distance": 0.021473845466971397,
            "std_dev_linear_probe_accuracy": 0,
            "pearson_image_intermodality_rsa": 0.8790264733692879,
            "text_S6": 2.9339494705200195,
            "image_S0": 11.47271728515625,
            "uniformity": -1.580543041229248,
            "text_variance0": 0.2528659701347351,
            "average_linear_probe_accuracy": -1,
            "text_S7": 2.5983593463897705,
            "full_text_rank": 16,
            "image_variance5": 0.08164919912815094,
            "linear_seperability_accuracy": 0.4902439024390244,
            "text_variance4": 0.0929962992668152,
            "train_cyclic_loss": 2.3437529296875024,
            "train_uniform_cyclic_loss": 0.03676041960716248,
            "cifar10_mean_cosine_similarity": 0.8188523054122925,
            "text_S0": 11.327112197875977,
            "image_S5": 3.052720069885254,
            "pearson_text_intermodality_rsa": 0.8796252571769817,
            "cifar10_centroid_euclidean_distance": 0.1222427263855934,
            "train_rsa_loss": -100,
            "image_S1": 4.854349613189697,
            "image_S2": 4.310520648956299,
            "image_S6": 2.928680896759033,
            "text_variance5": 0.08165622502565384,
            "mean_pairwise_euclidean_distance": 0.3295612335205078,
            "train_cyclic_dir_loss": 0.01465468853712082,
            "train_intramodality_loss": -100,
            "centroid_cosine_similarity": 0.9996109008789062,
            "caltech101_linear_probe_accuracy": -1,
            "train_intermodality_loss": 1.573378086090088,
            "val_image_retrieval_accuracy": 0.62109375,
            "val_image_classification_accuracy": 0.619140625,
            "cifar10_val_image_classification_accuracy": 0.7472,
            "cifar100_val_image_classification_accuracy": 0.3088,
            "first_lt1_value": -1,
            "image_variance4": 0.09127018600702286,
            "image_variance6": 0.06883949786424637,
            "train_alignment_loss": 0.11769862473011015,
            "train_pearson_loss": -100,
            "mean_cosine_similarity": 0.9411506652832032,
            "mean_text_text_cosine_similarity": 0.5625340938568115,
            "average_intra_modality_cosine_similarity": 0.565917581319809,
            "_step": 11000,
            "text_S1": 4.925261974334717,
            "image_S3": 3.882675647735596,
            "text_variance1": 0.1939789354801178,
            "_runtime": 8075.636269330978,
            "_timestamp": 1716179159.8391333,
            "text_variance6": 0.06853871047496796,
            "image_variance3": 0.11438021808862686,
            "avg_S": 4.566193580627441,
            "_wandb.runtime": 8112,
            "text_S2": 4.361648082733154,
            "text_S3": 3.915069341659546,
            "text_intermodality_rsa": 0.8696794735617206,
            "dtd_linear_probe_accuracy": -1,
            "mean_image_image_cosine_similarity": 0.5693010687828064,
            "svd": -100,
            "text_variance3": 0.12049806118011476,
            "text_variance7": 0.03747089207172394,
            "image_variance2": 0.14940528571605682,
            "rsa_before_interchanging": 0.7556060050198715,
            "cifar10_linear_probe_accuracy": 0.9004,
            "text_S4": 3.346011161804199,
            "image_rank": 16,
            "full_image_rank": 16,
            "train_total_loss": 1.573378086090088,
            "text_S5": 3.122138738632202,
            "image_intermodality_rsa": 0.8693459700044968,
            "train_cross_uniformity_loss": -1.5815982818603516,
            "cifar10_image_uniformity_loss": -0.7528746724128723,
            "cifar10_temp_scaled_inter_modality_loss": 0.9547439813613892,
            "image_S4": 3.2542946338653564,
            "image_S7": 2.5167274475097656,
            "text_rank": 16,
            "cifar10_inter_modality_loss": 2.1864047050476074
            },
        'config': {
            "lr": {
                "desc": None,
                "value": 0.00001
            },
            "host": {
                "desc": None,
                "value": "cirrus"
            },
            "seed": {
                "desc": None,
                "value": 42
            },
            "_wandb": {
                "desc": None,
                "value": {
                "t": {
                    "1": [
                    1,
                    5,
                    11,
                    41,
                    49,
                    51,
                    53,
                    55,
                    100
                    ],
                    "2": [
                    1,
                    5,
                    11,
                    41,
                    49,
                    51,
                    53,
                    55,
                    100
                    ],
                    "3": [
                    2,
                    23,
                    37
                    ],
                    "4": "3.10.13",
                    "5": "0.16.0",
                    "6": "4.34.0",
                    "8": [
                    5
                    ],
                    "13": "linux-x86_64"
                },
                "framework": "huggingface",
                "start_time": 1716171084.202864,
                "cli_version": "0.16.0",
                "is_jupyter_run": False,
                "python_version": "3.10.13",
                "is_kaggle_kernel": False,
                "huggingface_version": "4.34.0"
                }
            },
            "dataset": {
                "desc": None,
                "value": "mscoco"
            },
            "csv_path": {
                "desc": None,
                "value": "stats/"
            },
            "n_epochs": {
                "desc": None,
                "value": 6
            },
            "rsa_loss": {
                "desc": None,
                "value": False
            },
            "svd_loss": {
                "desc": None,
                "value": False
            },
            "max_steps": {
                "desc": None,
                "value": None
            },
            "batch_size": {
                "desc": None,
                "value": 64
            },
            "grad_cache": {
                "desc": None,
                "value": False
            },
            "save_every": {
                "desc": None,
                "value": 200
            },
            "W_layer_gap": {
                "desc": None,
                "value": -1
            },
            "cifar10_acc": {
                "desc": None,
                "value": True
            },
            "cuda_device": {
                "desc": None,
                "value": "cuda:0"
            },
            "cyclip_loss": {
                "desc": None,
                "value": False
            },
            "num_workers": {
                "desc": None,
                "value": 12
            },
            "one_encoder": {
                "desc": None,
                "value": False
            },
            "same_inputs": {
                "desc": None,
                "value": False
            },
            "save_losses": {
                "desc": None,
                "value": False
            },
            "simclr_loss": {
                "desc": None,
                "value": False
            },
            "temperature": {
                "desc": None,
                "value": 0.01
            },
            "loss_weights": {
                "desc": None,
                "value": {
                "image_to_text_weight": 0.5,
                "text_to_image_weight": 0.5
                }
            },
            "pearson_loss": {
                "desc": None,
                "value": False
            },
            "same_encoder": {
                "desc": None,
                "value": False
            },
            "vision_model": {
                "desc": None,
                "value": "VIT"
            },
            "weight_decay": {
                "desc": None,
                "value": 0.1
            },
            "hf_clip_model": {
                "desc": None,
                "value": "openai/clip-vit-base-patch32"
            },
            "use_scheduler": {
                "desc": None,
                "value": "EXP"
            },
            "alignment_loss": {
                "desc": None,
                "value": False
            },
            "n_warmup_steps": {
                "desc": None,
                "value": 10000
            },
            "schedule_every": {
                "desc": None,
                "value": 400
            },
            "uniformity_loss": {
                "desc": None,
                "value": False
            },
            "cifar_batch_size": {
                "desc": None,
                "value": 128
            },
            "do_checkpointing": {
                "desc": None,
                "value": True
            },
            "mismatched_pairs": {
                "desc": None,
                "value": False
            },
            "n_embeds_to_save": {
                "desc": None,
                "value": 512
            },
            "use_train_as_val": {
                "desc": None,
                "value": False
            },
            "cosine_align_loss": {
                "desc": None,
                "value": False
            },
            "encoder1_modality": {
                "desc": None,
                "value": "image"
            },
            "encoder2_modality": {
                "desc": None,
                "value": "text"
            },
            "openai_clip_model": {
                "desc": None,
                "value": "ViT-B/32"
            },
            "scaled_denominator": {
                "desc": None,
                "value": False
            },
            "train_from_scratch": {
                "desc": None,
                "value": False
            },
            "clip_projection_dim": {
                "desc": None,
                "value": 16
            },
            "intra_modality_loss": {
                "desc": None,
                "value": False
            },
            "selected_clip_model": {
                "desc": None,
                "value": "clip_finetuned_temp"
            },
            "uniform_cyclic_loss": {
                "desc": None,
                "value": False
            },
            "train_only_one_batch": {
                "desc": None,
                "value": False
            },
            "use_cached_val_batch": {
                "desc": None,
                "value": True
            },
            "visualize_embeddings": {
                "desc": None,
                "value": False
            },
            "cross_uniformity_loss": {
                "desc": None,
                "value": False
            },
            "cyclic_direction_loss": {
                "desc": None,
                "value": False
            },
            "grad_cache_multiplier": {
                "desc": None,
                "value": 16
            },
            "learnable_temperature": {
                "desc": None,
                "value": False
            },
            "second_caption_offset": {
                "desc": None,
                "value": False
            },
            "show_incorrect_images": {
                "desc": None,
                "value": False
            },
            "train_from_pretrained": {
                "desc": None,
                "value": True
            },
            "use_small_trainloader": {
                "desc": None,
                "value": False
            },
            "validation_batch_size": {
                "desc": None,
                "value": 512
            },
            "cosine_uniformity_loss": {
                "desc": None,
                "value": False
            },
            "delete_val_batch_first": {
                "desc": None,
                "value": False
            },
            "finetune_clip_backbone": {
                "desc": None,
                "value": True
            },
            "common_projection_layer": {
                "desc": None,
                "value": False
            },
            "loss_file_name_template": {
                "desc": None,
                "value": "Ttemp_loss_seed_trainmode_captionencoder_dim_val_bsize_dataset_vmodel_pretrained_FINAL"
            },
            "remove_contrastive_loss": {
                "desc": None,
                "value": False
            },
            "validation_dataset_size": {
                "desc": None,
                "value": 512
            },
            "continue_from_checkpoint": {
                "desc": None,
                "value": False
            },
            "shared_transformer_layers": {
                "desc": None,
                "value": False
            },
            "zero_shot_acc_num_workers": {
                "desc": None,
                "value": 4
            },
            "intra_modality_temperature": {
                "desc": None,
                "value": 0.01
            },
            "save_encoder_hidden_states": {
                "desc": None,
                "value": False
            },
            "finetune_multi_layer_projection": {
                "desc": None,
                "value": False
            }
            }
    },

    {
        'summary': {
            "cifar10_temp_scaled_inter_modality_loss": 3.085880994796753,
            "text_S2": 6.053715705871582,
            "uniformity": -3.383345603942871,
            "image_variance4": 0.12029461562633514,
            "dtd_linear_probe_accuracy": -1,
            "cifar10_inter_modality_loss": 1.8758617639541624,
            "pearson_text_intermodality_rsa": 0.8760314551535583,
            "svd": -100,
            "text_S4": 5.486788749694824,
            "image_S5": 5.24686336517334,
            "first_lt1_value": -1,
            "text_S6": 5.005610466003418,
            "image_S0": 6.679508209228516,
            "text_variance6": 0.09754758328199388,
            "image_variance5": 0.10778608918190002,
            "_wandb.runtime": 9769,
            "text_S3": 5.667654037475586,
            "image_S1": 6.303207874298096,
            "image_S2": 5.960450172424316,
            "text_rank": 16,
            "temperature": 0.010000000000000004,
            "train_cyclic_loss": 7.674663671875007,
            "cifar100_linear_probe_accuracy": 0.6884,
            "image_variance6": 0.09418290853500366,
            "mean_cosine_similarity": 0.8714017271995544,
            "centroid_euclidean_distance": 0.03633347898721695,
            "linear_seperability_accuracy": 0.5,
            "cifar10_linear_probe_accuracy": 0.9054,
            "caltech101_linear_probe_accuracy": -1,
            "text_S7": 4.508505821228027,
            "image_S3": 5.7574543952941895,
            "image_variance0": 0.17398351430892944,
            "train_alignment_loss": 0.2571965157985687,
            "text_intermodality_rsa": 0.8550221280616611,
            "cifar10_image_uniformity_loss": -2.07892107963562,
            "_runtime": 9736.8815677166,
            "_timestamp": 1716180935.3187995,
            "image_variance7": 0.08019034564495087,
            "train_cross_uniformity_loss": -3.387212038040161,
            "val_image_classification_accuracy": 0.595703125,
            "mean_pairwise_euclidean_distance": 0.47154852747917175,
            "non_similar_mean_cosine_similarity": -0.0001532065507490188,
            "avg_S": 5.618385314941406,
            "text_S1": 6.317570686340332,
            "text_S5": 5.293122291564941,
            "full_text_rank": 16,
            "text_variance5": 0.109646737575531,
            "val_image_retrieval_accuracy": 0.607421875,
            "_step": 11000,
            "image_rank": 16,
            "text_variance7": 0.07956518232822418,
            "train_rsa_loss": -100,
            "train_uniformity_loss": -3.383345603942871,
            "rsa_before_interchanging": 0.7291699577889543,
            "mean_text_text_cosine_similarity": -0.00013312470400705934,
            "pearson_rsa_before_interchanging": 0.7651577661411532,
            "text_S0": 6.614114284515381,
            "text_variance2": 0.1433643102645874,
            "image_variance2": 0.13906332850456238,
            "train_pearson_loss": -100,
            "cifar10_mean_cosine_similarity": 0.33732813596725464,
            "pearson_image_intermodality_rsa": 0.8720371921712311,
            "text_variance4": 0.1175788789987564,
            "image_intermodality_rsa": 0.8507992357367498,
            "train_uniform_cyclic_loss": 0.12038721889257432,
            "full_image_rank": 16,
            "train_intramodality_loss": -100,
            "average_intra_modality_cosine_similarity": 0.00025648600421845913,
            "cifar10_val_image_classification_accuracy": 0.7767,
            "cifar100_val_image_classification_accuracy": 0.318,
            "image_S4": 5.546576023101807,
            "image_S7": 4.524312973022461,
            "text_variance0": 0.17094111442565918,
            "text_variance1": 0.155815452337265,
            "image_variance1": 0.1550661474466324,
            "cifar10_centroid_euclidean_distance": 0.21684524416923523,
            "image_S6": 4.915581703186035,
            "image_variance3": 0.12943309545516968,
            "train_total_loss": 0.28043946623802185,
            "centroid_cosine_similarity": 0.7123315930366516,
            "std_dev_linear_probe_accuracy": 0,
            "mean_image_image_cosine_similarity": 0.0006460967124439776,
            "text_variance3": 0.12554077804088593,
            "train_cyclic_dir_loss": 0.03198454901576042,
            "train_intermodality_loss": 3.406588554382324,
            "average_linear_probe_accuracy": -1
            },
        'config': {
                "lr": {
                    "desc": None,
                    "value": 0.00001
                },
                "host": {
                    "desc": None,
                    "value": "cirrus"
                },
                "seed": {
                    "desc": None,
                    "value": 42
                },
                "_wandb": {
                    "desc": None,
                    "value": {
                    "t": {
                        "1": [
                        1,
                        5,
                        11,
                        41,
                        49,
                        51,
                        53,
                        55,
                        100
                        ],
                        "2": [
                        1,
                        5,
                        11,
                        41,
                        49,
                        51,
                        53,
                        55,
                        100
                        ],
                        "3": [
                        2,
                        23,
                        37
                        ],
                        "4": "3.10.13",
                        "5": "0.16.0",
                        "6": "4.34.0",
                        "8": [
                        5
                        ],
                        "13": "linux-x86_64"
                    },
                    "framework": "huggingface",
                    "start_time": 1716171198.437232,
                    "cli_version": "0.16.0",
                    "is_jupyter_run": False,
                    "python_version": "3.10.13",
                    "is_kaggle_kernel": False,
                    "huggingface_version": "4.34.0"
                    }
                },
                "dataset": {
                    "desc": None,
                    "value": "mscoco"
                },
                "csv_path": {
                    "desc": None,
                    "value": "stats/"
                },
                "n_epochs": {
                    "desc": None,
                    "value": 6
                },
                "rsa_loss": {
                    "desc": None,
                    "value": False
                },
                "svd_loss": {
                    "desc": None,
                    "value": False
                },
                "max_steps": {
                    "desc": None,
                    "value": None
                },
                "batch_size": {
                    "desc": None,
                    "value": 64
                },
                "grad_cache": {
                    "desc": None,
                    "value": False
                },
                "save_every": {
                    "desc": None,
                    "value": 200
                },
                "W_layer_gap": {
                    "desc": None,
                    "value": -1
                },
                "cifar10_acc": {
                    "desc": None,
                    "value": True
                },
                "cuda_device": {
                    "desc": None,
                    "value": "cuda:1"
                },
                "cyclip_loss": {
                    "desc": None,
                    "value": False
                },
                "num_workers": {
                    "desc": None,
                    "value": 12
                },
                "one_encoder": {
                    "desc": None,
                    "value": False
                },
                "same_inputs": {
                    "desc": None,
                    "value": False
                },
                "save_losses": {
                    "desc": None,
                    "value": False
                },
                "simclr_loss": {
                    "desc": None,
                    "value": False
                },
                "temperature": {
                    "desc": None,
                    "value": 0.01
                },
                "loss_weights": {
                    "desc": None,
                    "value": {
                    "image_to_text_weight": 0.5,
                    "text_to_image_weight": 0.5
                    }
                },
                "pearson_loss": {
                    "desc": None,
                    "value": False
                },
                "same_encoder": {
                    "desc": None,
                    "value": False
                },
                "vision_model": {
                    "desc": None,
                    "value": "VIT"
                },
                "weight_decay": {
                    "desc": None,
                    "value": 0.1
                },
                "hf_clip_model": {
                    "desc": None,
                    "value": "openai/clip-vit-base-patch32"
                },
                "use_scheduler": {
                    "desc": None,
                    "value": "EXP"
                },
                "alignment_loss": {
                    "desc": None,
                    "value": True
                },
                "n_warmup_steps": {
                    "desc": None,
                    "value": 10000
                },
                "schedule_every": {
                    "desc": None,
                    "value": 400
                },
                "uniformity_loss": {
                    "desc": None,
                    "value": True
                },
                "cifar_batch_size": {
                    "desc": None,
                    "value": 128
                },
                "do_checkpointing": {
                    "desc": None,
                    "value": True
                },
                "mismatched_pairs": {
                    "desc": None,
                    "value": False
                },
                "n_embeds_to_save": {
                    "desc": None,
                    "value": 512
                },
                "use_train_as_val": {
                    "desc": None,
                    "value": False
                },
                "cosine_align_loss": {
                    "desc": None,
                    "value": False
                },
                "encoder1_modality": {
                    "desc": None,
                    "value": "image"
                },
                "encoder2_modality": {
                    "desc": None,
                    "value": "text"
                },
                "openai_clip_model": {
                    "desc": None,
                    "value": "ViT-B/32"
                },
                "scaled_denominator": {
                    "desc": None,
                    "value": False
                },
                "train_from_scratch": {
                    "desc": None,
                    "value": False
                },
                "clip_projection_dim": {
                    "desc": None,
                    "value": 16
                },
                "intra_modality_loss": {
                    "desc": None,
                    "value": False
                },
                "selected_clip_model": {
                    "desc": None,
                    "value": "clip_finetuned_temp"
                },
                "uniform_cyclic_loss": {
                    "desc": None,
                    "value": False
                },
                "train_only_one_batch": {
                    "desc": None,
                    "value": False
                },
                "use_cached_val_batch": {
                    "desc": None,
                    "value": True
                },
                "visualize_embeddings": {
                    "desc": None,
                    "value": False
                },
                "cross_uniformity_loss": {
                    "desc": None,
                    "value": False
                },
                "cyclic_direction_loss": {
                    "desc": None,
                    "value": False
                },
                "grad_cache_multiplier": {
                    "desc": None,
                    "value": 16
                },
                "learnable_temperature": {
                    "desc": None,
                    "value": False
                },
                "second_caption_offset": {
                    "desc": None,
                    "value": False
                },
                "show_incorrect_images": {
                    "desc": None,
                    "value": False
                },
                "train_from_pretrained": {
                    "desc": None,
                    "value": True
                },
                "use_small_trainloader": {
                    "desc": None,
                    "value": False
                },
                "validation_batch_size": {
                    "desc": None,
                    "value": 512
                },
                "cosine_uniformity_loss": {
                    "desc": None,
                    "value": False
                },
                "delete_val_batch_first": {
                    "desc": None,
                    "value": False
                },
                "finetune_clip_backbone": {
                    "desc": None,
                    "value": True
                },
                "common_projection_layer": {
                    "desc": None,
                    "value": False
                },
                "loss_file_name_template": {
                    "desc": None,
                    "value": "Ttemp_loss_seed_trainmode_captionencoder_dim_val_bsize_dataset_vmodel_pretrained_FINAL"
                },
                "remove_contrastive_loss": {
                    "desc": None,
                    "value": False
                },
                "validation_dataset_size": {
                    "desc": None,
                    "value": 512
                },
                "continue_from_checkpoint": {
                    "desc": None,
                    "value": False
                },
                "shared_transformer_layers": {
                    "desc": None,
                    "value": False
                },
                "zero_shot_acc_num_workers": {
                    "desc": None,
                    "value": 4
                },
                "intra_modality_temperature": {
                    "desc": None,
                    "value": 0.01
                },
                "save_encoder_hidden_states": {
                    "desc": None,
                    "value": False
                },
                "finetune_multi_layer_projection": {
                    "desc": None,
                    "value": False
                }
                }
    },


    {
        # 64D CLIP
        'summary': {
            "image_S0": 5.298120498657227,
            "temperature": 0.010000000000000004,
            "text_variance2": 0.13438494503498075,
            "text_variance6": 0.016010865569114685,
            "train_uniformity_loss": -1.6356980800628662,
            "_wandb.runtime": 8783,
            "text_S2": 2.0020127296447754,
            "image_S2": 1.9166300296783447,
            "image_intermodality_rsa": 0.7810457093722973,
            "cifar10_linear_probe_accuracy": 0.9084,
            "image_S3": 1.466128706932068,
            "image_S4": 1.1715130805969238,
            "centroid_cosine_similarity": 0.9462392330169678,
            "pearson_text_intermodality_rsa": 0.8186489053136927,
            "mean_image_image_cosine_similarity": 0.5891945958137512,
            "cifar10_temp_scaled_inter_modality_loss": 0.7960696220397949,
            "image_rank": 41,
            "text_variance3": 0.08123276382684708,
            "text_variance4": 0.04898339882493019,
            "text_variance5": 0.02687590569257736,
            "image_variance0": 0.46854427456855774,
            "train_total_loss": 1.289703607559204,
            "mean_text_text_cosine_similarity": 0.5556365251541138,
            "non_similar_mean_cosine_similarity": 0.5415878295898438,
            "image_variance4": 0.04968678578734398,
            "mean_cosine_similarity": 0.8553823828697205,
            "rsa_before_interchanging": 0.627238992656837,
            "train_intermodality_loss": 1.289703607559204,
            "cifar10_inter_modality_loss": 2.2046449184417725,
            "mean_pairwise_euclidean_distance": 0.5337882041931152,
            "image_S7": 0.5066552758216858,
            "text_rank": 41,
            "image_variance2": 0.13273699581623075,
            "cifar10_mean_cosine_similarity": 0.7608008980751038,
            "uniformity": -1.6356980800628662,
            "full_image_rank": 64,
            "train_cyclic_loss": 1.6099858398437514,
            "train_alignment_loss": 0.2892352044582367,
            "centroid_euclidean_distance": 0.249198317527771,
            "average_linear_probe_accuracy": -1,
            "full_text_rank": 64,
            "image_variance1": 0.21748754382133484,
            "text_intermodality_rsa": 0.7960489110189808,
            "dtd_linear_probe_accuracy": -1,
            "cifar10_val_image_classification_accuracy": 0.779,
            "cifar100_val_image_classification_accuracy": 0.3533,
            "svd": -100,
            "pearson_rsa_before_interchanging": 0.6684574507400891,
            "val_image_classification_accuracy": 0.671875,
            "avg_S": 1.843231439590454,
            "image_variance6": 0.016566487029194832,
            "train_pearson_loss": -100,
            "train_cross_uniformity_loss": -1.7636101245880127,
            "cifar100_linear_probe_accuracy": 0.6634,
            "caltech101_linear_probe_accuracy": -1,
            "text_S4": 1.2125844955444336,
            "text_variance0": 0.46951574087142944,
            "first_lt1_value": 0.959197461605072,
            "image_variance3": 0.07718412578105927,
            "train_cyclic_dir_loss": 0.007097981404513121,
            "val_image_retrieval_accuracy": 0.6875,
            "std_dev_linear_probe_accuracy": 0,
            "text_S3": 1.5612773895263672,
            "text_S5": 0.8967400789260864,
            "text_S6": 0.6774754524230957,
            "text_S7": 0.4905846416950226,
            "text_variance1": 0.21455679833889008,
            "image_variance5": 0.0282245259732008,
            "_step": 11000,
            "text_S0": 5.372421741485596,
            "image_S6": 0.6734693050384521,
            "text_variance7": 0.008439578115940094,
            "image_variance7": 0.009569224901497364,
            "cifar10_centroid_euclidean_distance": 0.3599262833595276,
            "_runtime": 8748.596397638321,
            "image_S5": 0.8878350257873535,
            "train_uniform_cyclic_loss": 0.0283429604023695,
            "cifar10_image_uniformity_loss": -0.8831652998924255,
            "text_S1": 2.532754421234131,
            "image_S1": 2.454017639160156,
            "_timestamp": 1716179887.8613143,
            "train_intramodality_loss": -100,
            "train_rsa_loss": -100,
            "linear_seperability_accuracy": 0.996341463414634,
            "pearson_image_intermodality_rsa": 0.8069300909573313,
            "average_intra_modality_cosine_similarity": 0.5724155604839325
            },
        'config': {
            "lr": {
                "desc": None,
                "value": 0.00001
            },
            "host": {
                "desc": None,
                "value": "cirrus"
            },
            "seed": {
                "desc": None,
                "value": 42
            },
            "_wandb": {
                "desc": None,
                "value": {
                "t": {
                    "1": [
                    1,
                    5,
                    11,
                    41,
                    49,
                    51,
                    53,
                    55,
                    100
                    ],
                    "2": [
                    1,
                    5,
                    11,
                    41,
                    49,
                    51,
                    53,
                    55,
                    100
                    ],
                    "3": [
                    2,
                    23,
                    37
                    ],
                    "4": "3.10.13",
                    "5": "0.16.0",
                    "6": "4.34.0",
                    "8": [
                    5
                    ],
                    "13": "linux-x86_64"
                },
                "framework": "huggingface",
                "start_time": 1716171139.264917,
                "cli_version": "0.16.0",
                "is_jupyter_run": False,
                "python_version": "3.10.13",
                "is_kaggle_kernel": False,
                "huggingface_version": "4.34.0"
                }
            },
            "dataset": {
                "desc": None,
                "value": "mscoco"
            },
            "csv_path": {
                "desc": None,
                "value": "stats/"
            },
            "n_epochs": {
                "desc": None,
                "value": 6
            },
            "rsa_loss": {
                "desc": None,
                "value": False
            },
            "svd_loss": {
                "desc": None,
                "value": False
            },
            "max_steps": {
                "desc": None,
                "value": None
            },
            "batch_size": {
                "desc": None,
                "value": 64
            },
            "grad_cache": {
                "desc": None,
                "value": False
            },
            "save_every": {
                "desc": None,
                "value": 200
            },
            "W_layer_gap": {
                "desc": None,
                "value": -1
            },
            "cifar10_acc": {
                "desc": None,
                "value": True
            },
            "cuda_device": {
                "desc": None,
                "value": "cuda:0"
            },
            "cyclip_loss": {
                "desc": None,
                "value": False
            },
            "num_workers": {
                "desc": None,
                "value": 12
            },
            "one_encoder": {
                "desc": None,
                "value": False
            },
            "same_inputs": {
                "desc": None,
                "value": False
            },
            "save_losses": {
                "desc": None,
                "value": False
            },
            "simclr_loss": {
                "desc": None,
                "value": False
            },
            "temperature": {
                "desc": None,
                "value": 0.01
            },
            "loss_weights": {
                "desc": None,
                "value": {
                "image_to_text_weight": 0.5,
                "text_to_image_weight": 0.5
                }
            },
            "pearson_loss": {
                "desc": None,
                "value": False
            },
            "same_encoder": {
                "desc": None,
                "value": False
            },
            "vision_model": {
                "desc": None,
                "value": "VIT"
            },
            "weight_decay": {
                "desc": None,
                "value": 0.1
            },
            "hf_clip_model": {
                "desc": None,
                "value": "openai/clip-vit-base-patch32"
            },
            "use_scheduler": {
                "desc": None,
                "value": "EXP"
            },
            "alignment_loss": {
                "desc": None,
                "value": False
            },
            "n_warmup_steps": {
                "desc": None,
                "value": 10000
            },
            "schedule_every": {
                "desc": None,
                "value": 400
            },
            "uniformity_loss": {
                "desc": None,
                "value": False
            },
            "cifar_batch_size": {
                "desc": None,
                "value": 128
            },
            "do_checkpointing": {
                "desc": None,
                "value": True
            },
            "mismatched_pairs": {
                "desc": None,
                "value": False
            },
            "n_embeds_to_save": {
                "desc": None,
                "value": 512
            },
            "use_train_as_val": {
                "desc": None,
                "value": False
            },
            "cosine_align_loss": {
                "desc": None,
                "value": False
            },
            "encoder1_modality": {
                "desc": None,
                "value": "image"
            },
            "encoder2_modality": {
                "desc": None,
                "value": "text"
            },
            "openai_clip_model": {
                "desc": None,
                "value": "ViT-B/32"
            },
            "scaled_denominator": {
                "desc": None,
                "value": False
            },
            "train_from_scratch": {
                "desc": None,
                "value": False
            },
            "clip_projection_dim": {
                "desc": None,
                "value": 64
            },
            "intra_modality_loss": {
                "desc": None,
                "value": False
            },
            "selected_clip_model": {
                "desc": None,
                "value": "clip_finetuned_temp"
            },
            "uniform_cyclic_loss": {
                "desc": None,
                "value": False
            },
            "train_only_one_batch": {
                "desc": None,
                "value": False
            },
            "use_cached_val_batch": {
                "desc": None,
                "value": True
            },
            "visualize_embeddings": {
                "desc": None,
                "value": False
            },
            "cross_uniformity_loss": {
                "desc": None,
                "value": False
            },
            "cyclic_direction_loss": {
                "desc": None,
                "value": False
            },
            "grad_cache_multiplier": {
                "desc": None,
                "value": 16
            },
            "learnable_temperature": {
                "desc": None,
                "value": False
            },
            "second_caption_offset": {
                "desc": None,
                "value": False
            },
            "show_incorrect_images": {
                "desc": None,
                "value": False
            },
            "train_from_pretrained": {
                "desc": None,
                "value": True
            },
            "use_small_trainloader": {
                "desc": None,
                "value": False
            },
            "validation_batch_size": {
                "desc": None,
                "value": 512
            },
            "cosine_uniformity_loss": {
                "desc": None,
                "value": False
            },
            "delete_val_batch_first": {
                "desc": None,
                "value": False
            },
            "finetune_clip_backbone": {
                "desc": None,
                "value": True
            },
            "common_projection_layer": {
                "desc": None,
                "value": False
            },
            "loss_file_name_template": {
                "desc": None,
                "value": "Ttemp_loss_seed_trainmode_captionencoder_dim_val_bsize_dataset_vmodel_pretrained_FINAL"
            },
            "remove_contrastive_loss": {
                "desc": None,
                "value": False
            },
            "validation_dataset_size": {
                "desc": None,
                "value": 512
            },
            "continue_from_checkpoint": {
                "desc": None,
                "value": False
            },
            "shared_transformer_layers": {
                "desc": None,
                "value": False
            },
            "zero_shot_acc_num_workers": {
                "desc": None,
                "value": 4
            },
            "intra_modality_temperature": {
                "desc": None,
                "value": 0.01
            },
            "save_encoder_hidden_states": {
                "desc": None,
                "value": False
            },
            "finetune_multi_layer_projection": {
                "desc": None,
                "value": False
            }
            }
    },
    {
        'summary': {
            "image_variance4": 0.05147012695670128,
            "image_variance5": 0.028152653947472572,
            "image_variance7": 0.008626212365925312,
            "linear_seperability_accuracy": 0.5073170731707317,
            "image_S2": 3.2211320400238037,
            "uniformity": -3.63630485534668,
            "train_rsa_loss": -100,
            "avg_S": 2.4338324069976807,
            "text_S5": 1.3101298809051514,
            "text_S7": 0.7023578882217407,
            "image_S5": 1.3351781368255615,
            "text_variance7": 0.007803826592862606,
            "svd": -100,
            "image_S1": 3.992382526397705,
            "image_S6": 0.9889044761657716,
            "text_variance1": 0.246175080537796,
            "text_variance6": 0.014127553440630436,
            "rsa_before_interchanging": 0.6263057537336604,
            "cifar10_inter_modality_loss": 1.9039623737335205,
            "caltech101_linear_probe_accuracy": -1,
            "mean_text_text_cosine_similarity": 0.00013043924991507083,
            "first_lt1_value": 0.9567292332649232,
            "train_cyclic_loss": 4.261853515625004,
            "text_S3": 2.4953460693359375,
            "val_image_retrieval_accuracy": 0.6484375,
            "std_dev_linear_probe_accuracy": 0,
            "mean_pairwise_euclidean_distance": 0.6150933504104614,
            "cifar10_linear_probe_accuracy": 0.9046,
            "mean_image_image_cosine_similarity": 0.0014085130533203485,
            "average_intra_modality_cosine_similarity": 0.0007694761516177095,
            "_wandb.runtime": 8784,
            "text_S0": 4.965303897857666,
            "dtd_linear_probe_accuracy": -1,
            "centroid_cosine_similarity": 0.6897773742675781,
            "pearson_rsa_before_interchanging": 0.723994982383356,
            "cifar10_val_image_classification_accuracy": 0.7981,
            "cifar100_val_image_classification_accuracy": 0.3635,
            "image_variance1": 0.25023025274276733,
            "image_variance2": 0.16313596069812775,
            "train_total_loss": -0.15881967544555664,
            "train_intermodality_loss": 3.0726025104522705,
            "train_uniform_cyclic_loss": 0.06678057461977005,
            "average_linear_probe_accuracy": -1,
            "cifar10_mean_cosine_similarity": 0.2739006578922272,
            "cifar10_temp_scaled_inter_modality_loss": 2.5020925998687744,
            "image_S0": 4.970821380615234,
            "text_variance0": 0.3873473107814789,
            "image_variance0": 0.3875432908535004,
            "train_cross_uniformity_loss": -3.647756099700928,
            "val_image_classification_accuracy": 0.638671875,
            "text_S1": 3.964444160461426,
            "text_variance2": 0.16737449169158936,
            "text_variance4": 0.05197184532880783,
            "train_pearson_loss": -100,
            "train_alignment_loss": 0.40488266944885254,
            "text_S4": 1.817919135093689,
            "text_S6": 0.9476362466812134,
            "text_variance3": 0.0979910045862198,
            "text_variance5": 0.027208909392356873,
            "full_image_rank": 64,
            "train_cyclic_dir_loss": 0.012596556916832924,
            "train_uniformity_loss": -3.63630485534668,
            "train_intramodality_loss": -100,
            "cifar10_image_uniformity_loss": -2.273869037628174,
            "image_S4": 1.8081984519958496,
            "text_rank": 51,
            "_timestamp": 1716179969.9280572,
            "image_variance6": 0.01532598864287138,
            "image_intermodality_rsa": 0.7844493151994723,
            "text_S2": 3.2675223350524902,
            "image_S3": 2.46236252784729,
            "temperature": 0.010000000000000004,
            "centroid_euclidean_distance": 0.04234221577644348,
            "cifar10_centroid_euclidean_distance": 0.45679759979248047,
            "image_S7": 0.7398295402526855,
            "mean_cosine_similarity": 0.7975587248802185,
            "non_similar_mean_cosine_similarity": 0.0002674572169780731,
            "_step": 11000,
            "_runtime": 8750.529324293137,
            "image_rank": 51,
            "full_text_rank": 64,
            "image_variance3": 0.09551554173231123,
            "text_intermodality_rsa": 0.7898580255779967,
            "cifar100_linear_probe_accuracy": 0.6766,
            "pearson_text_intermodality_rsa": 0.8490446254094055,
            "pearson_image_intermodality_rsa": 0.8429207218705499
            },
        'config': {
            "lr": {
                "desc": None,
                "value": 0.00001
            },
            "host": {
                "desc": None,
                "value": "cirrus"
            },
            "seed": {
                "desc": None,
                "value": 42
            },
            "dataset": {
                "desc": None,
                "value": "mscoco"
            },
            "csv_path": {
                "desc": None,
                "value": "stats/"
            },
            "n_epochs": {
                "desc": None,
                "value": 6
            },
            "rsa_loss": {
                "desc": None,
                "value": False
            },
            "svd_loss": {
                "desc": None,
                "value": False
            },
            "max_steps": {
                "desc": None,
                "value": None
            },
            "batch_size": {
                "desc": None,
                "value": 64
            },
            "grad_cache": {
                "desc": None,
                "value": False
            },
            "save_every": {
                "desc": None,
                "value": 200
            },
            "W_layer_gap": {
                "desc": None,
                "value": -1
            },
            "cifar10_acc": {
                "desc": None,
                "value": True
            },
            "cuda_device": {
                "desc": None,
                "value": "cuda:1"
            },
            "cyclip_loss": {
                "desc": None,
                "value": False
            },
            "num_workers": {
                "desc": None,
                "value": 12
            },
            "one_encoder": {
                "desc": None,
                "value": False
            },
            "same_inputs": {
                "desc": None,
                "value": False
            },
            "save_losses": {
                "desc": None,
                "value": False
            },
            "simclr_loss": {
                "desc": None,
                "value": False
            },
            "temperature": {
                "desc": None,
                "value": 0.01
            },
            "loss_weights": {
                "desc": None,
                "value": {
                "image_to_text_weight": 0.5,
                "text_to_image_weight": 0.5
                }
            },
            "pearson_loss": {
                "desc": None,
                "value": False
            },
            "same_encoder": {
                "desc": None,
                "value": False
            },
            "vision_model": {
                "desc": None,
                "value": "VIT"
            },
            "weight_decay": {
                "desc": None,
                "value": 0.1
            },
            "hf_clip_model": {
                "desc": None,
                "value": "openai/clip-vit-base-patch32"
            },
            "use_scheduler": {
                "desc": None,
                "value": "EXP"
            },
            "alignment_loss": {
                "desc": None,
                "value": True
            },
            "n_warmup_steps": {
                "desc": None,
                "value": 10000
            },
            "schedule_every": {
                "desc": None,
                "value": 400
            },
            "uniformity_loss": {
                "desc": None,
                "value": True
            },
            "cifar_batch_size": {
                "desc": None,
                "value": 128
            },
            "do_checkpointing": {
                "desc": None,
                "value": True
            },
            "mismatched_pairs": {
                "desc": None,
                "value": False
            },
            "n_embeds_to_save": {
                "desc": None,
                "value": 512
            },
            "use_train_as_val": {
                "desc": None,
                "value": False
            },
            "cosine_align_loss": {
                "desc": None,
                "value": False
            },
            "encoder1_modality": {
                "desc": None,
                "value": "image"
            },
            "encoder2_modality": {
                "desc": None,
                "value": "text"
            },
            "openai_clip_model": {
                "desc": None,
                "value": "ViT-B/32"
            },
            "scaled_denominator": {
                "desc": None,
                "value": False
            },
            "train_from_scratch": {
                "desc": None,
                "value": False
            },
            "clip_projection_dim": {
                "desc": None,
                "value": 64
            },
            "intra_modality_loss": {
                "desc": None,
                "value": False
            },
            "selected_clip_model": {
                "desc": None,
                "value": "clip_finetuned_temp"
            },
            "uniform_cyclic_loss": {
                "desc": None,
                "value": False
            },
            "train_only_one_batch": {
                "desc": None,
                "value": False
            },
            "use_cached_val_batch": {
                "desc": None,
                "value": True
            },
            "visualize_embeddings": {
                "desc": None,
                "value": False
            },
            "cross_uniformity_loss": {
                "desc": None,
                "value": False
            },
            "cyclic_direction_loss": {
                "desc": None,
                "value": False
            },
            "grad_cache_multiplier": {
                "desc": None,
                "value": 16
            },
            "learnable_temperature": {
                "desc": None,
                "value": False
            },
            "second_caption_offset": {
                "desc": None,
                "value": False
            },
            "show_incorrect_images": {
                "desc": None,
                "value": False
            },
            "train_from_pretrained": {
                "desc": None,
                "value": True
            },
            "use_small_trainloader": {
                "desc": None,
                "value": False
            },
            "validation_batch_size": {
                "desc": None,
                "value": 512
            },
            "cosine_uniformity_loss": {
                "desc": None,
                "value": False
            },
            "delete_val_batch_first": {
                "desc": None,
                "value": False
            },
            "finetune_clip_backbone": {
                "desc": None,
                "value": True
            },
            "common_projection_layer": {
                "desc": None,
                "value": False
            },
            "loss_file_name_template": {
                "desc": None,
                "value": "Ttemp_loss_seed_trainmode_captionencoder_dim_val_bsize_dataset_vmodel_pretrained_FINAL"
            },
            "remove_contrastive_loss": {
                "desc": None,
                "value": False
            },
            "validation_dataset_size": {
                "desc": None,
                "value": 512
            },
            "continue_from_checkpoint": {
                "desc": None,
                "value": False
            },
            "shared_transformer_layers": {
                "desc": None,
                "value": False
            },
            "zero_shot_acc_num_workers": {
                "desc": None,
                "value": 4
            },
            "intra_modality_temperature": {
                "desc": None,
                "value": 0.01
            },
            "save_encoder_hidden_states": {
                "desc": None,
                "value": False
            },
            "finetune_multi_layer_projection": {
                "desc": None,
                "value": False
            }
            }
    },
    # 128D CLIP 1e-6 no scheduling
    {
        'summary': {
            "image_S6": 0.40102338790893555,
            "text_variance0": 0.6325553059577942,
            "image_variance1": 0.18444129824638367,
            "mean_cosine_similarity": 0.59294193983078,
            "avg_S": 1.209261178970337,
            "image_S5": 0.5101279020309448,
            "image_variance0": 0.6332813501358032,
            "dtd_linear_probe_accuracy": -1,
            "std_dev_linear_probe_accuracy": 0,
            "pearson_text_intermodality_rsa": 0.786660857941639,
            "text_S4": 0.6339943408966064,
            "text_variance1": 0.1914898157119751,
            "train_cross_uniformity_loss": -2.778691053390503,
            "linear_seperability_accuracy": 1,
            "mean_image_image_cosine_similarity": 0.4813940823078155,
            "cifar10_centroid_euclidean_distance": 0.7398315668106079,
            "train_uniform_cyclic_loss": 0.03504985570907593,
            "cifar10_inter_modality_loss": 2.205206871032715,
            "image_S2": 1.1827338933944702,
            "image_S4": 0.659577488899231,
            "image_variance6": 0.009666569530963898,
            "train_alignment_loss": 0.8141161203384399,
            "cifar100_linear_probe_accuracy": -1,
            "pearson_image_intermodality_rsa": 0.7747844463794362,
            "temperature": 0.010000000000000004,
            "image_variance5": 0.015521777793765068,
            "train_cyclic_dir_loss": 0.00699261948466301,
            "text_intermodality_rsa": 0.7593935337024771,
            "non_similar_mean_cosine_similarity": 0.28873777389526367,
            "cifar10_temp_scaled_inter_modality_loss": 0.9268656373023988,
            "_step": 21200,
            "uniformity": -2.0137386322021484,
            "full_text_rank": 128,
            "image_variance4": 0.025890905410051342,
            "cifar10_val_image_classification_accuracy": 0.732,
            "image_S7": 0.2973349690437317,
            "image_rank": 47,
            "text_variance6": 0.007479526102542877,
            "train_rsa_loss": -100,
            "image_variance7": 0.005355256609618664,
            "mean_text_text_cosine_similarity": 0.4617941677570343,
            "mean_pairwise_euclidean_distance": 0.9012765884399414,
            "val_image_classification_accuracy": 0.634765625,
            "text_S2": 1.2276674509048462,
            "text_variance5": 0.013246778398752213,
            "first_lt1_value": 0.9992692470550536,
            "train_cyclic_loss": 1.8435578125000016,
            "image_intermodality_rsa": 0.7445773239202036,
            "train_intramodality_loss": -100,
            "text_S5": 0.4795847237110138,
            "train_intermodality_loss": 1.3993583917617798,
            "val_image_retrieval_accuracy": 0.640625,
            "_runtime": 12496.082690238953,
            "train_total_loss": 1.3993583917617798,
            "caltech101_linear_probe_accuracy": -1,
            "pearson_rsa_before_interchanging": 0.6430181454570734,
            "average_intra_modality_cosine_similarity": 0.4715941250324249,
            "_wandb.runtime": 12502,
            "image_variance3": 0.04386023432016373,
            "cifar10_image_uniformity_loss": -1.2085838317871094,
            "centroid_cosine_similarity": 0.6123104095458984,
            "centroid_euclidean_distance": 0.6054654717445374,
            "text_S1": 1.8466912508010864,
            "text_S3": 0.8622416257858276,
            "image_S0": 3.9697718620300297,
            "image_S1": 1.7852191925048828,
            "_timestamp": 1716210772.5309553,
            "image_variance2": 0.08198266476392746,
            "image_S3": 0.8628618717193604,
            "text_variance3": 0.04256545007228851,
            "train_pearson_loss": -100,
            "rsa_before_interchanging": 0.5930908362645194,
            "average_linear_probe_accuracy": -1,
            "cifar10_mean_cosine_similarity": 0.4951305091381073,
            "text_S0": 4.009056091308594,
            "text_S6": 0.35957810282707214,
            "text_variance4": 0.02303566597402096,
            "text_variance7": 0.003805792424827814,
            "full_image_rank": 128,
            "cifar10_linear_probe_accuracy": -1,
            "svd": -100,
            "text_S7": 0.2552759051322937,
            "text_rank": 49,
            "text_variance2": 0.08582174777984619,
            "train_uniformity_loss": -2.0137386322021484
            },
        'config': {
            "lr": {
                "desc": None,
                "value": 0.000001
            },
            "host": {
                "desc": None,
                "value": "cirrus"
            },
            "seed": {
                "desc": None,
                "value": 42
            },
            "_wandb": {
                "desc": None,
                "value": {
                "t": {
                    "1": [
                    1,
                    5,
                    11,
                    41,
                    49,
                    51,
                    53,
                    55,
                    100
                    ],
                    "2": [
                    1,
                    5,
                    11,
                    41,
                    49,
                    51,
                    53,
                    55,
                    100
                    ],
                    "3": [
                    23,
                    37
                    ],
                    "4": "3.10.13",
                    "5": "0.16.0",
                    "6": "4.34.0",
                    "8": [
                    5
                    ],
                    "13": "linux-x86_64"
                },
                "framework": "huggingface",
                "start_time": 1716198276.448265,
                "cli_version": "0.16.0",
                "is_jupyter_run": False,
                "python_version": "3.10.13",
                "is_kaggle_kernel": False,
                "huggingface_version": "4.34.0"
                }
            },
            "dataset": {
                "desc": None,
                "value": "mscoco"
            },
            "csv_path": {
                "desc": None,
                "value": "stats/"
            },
            "n_epochs": {
                "desc": None,
                "value": 20
            },
            "rsa_loss": {
                "desc": None,
                "value": False
            },
            "svd_loss": {
                "desc": None,
                "value": False
            },
            "max_steps": {
                "desc": None,
                "value": None
            },
            "batch_size": {
                "desc": None,
                "value": 64
            },
            "grad_cache": {
                "desc": None,
                "value": False
            },
            "save_every": {
                "desc": None,
                "value": 200
            },
            "W_layer_gap": {
                "desc": None,
                "value": -1
            },
            "cifar10_acc": {
                "desc": None,
                "value": True
            },
            "cuda_device": {
                "desc": None,
                "value": "cuda:0"
            },
            "cyclip_loss": {
                "desc": None,
                "value": False
            },
            "num_workers": {
                "desc": None,
                "value": 12
            },
            "one_encoder": {
                "desc": None,
                "value": False
            },
            "same_inputs": {
                "desc": None,
                "value": False
            },
            "save_losses": {
                "desc": None,
                "value": False
            },
            "simclr_loss": {
                "desc": None,
                "value": False
            },
            "temperature": {
                "desc": None,
                "value": 0.01
            },
            "loss_weights": {
                "desc": None,
                "value": {
                "image_to_text_weight": 0.5,
                "text_to_image_weight": 0.5
                }
            },
            "pearson_loss": {
                "desc": None,
                "value": False
            },
            "same_encoder": {
                "desc": None,
                "value": False
            },
            "vision_model": {
                "desc": None,
                "value": "VIT"
            },
            "weight_decay": {
                "desc": None,
                "value": 0.1
            },
            "hf_clip_model": {
                "desc": None,
                "value": "openai/clip-vit-base-patch32"
            },
            "use_scheduler": {
                "desc": None,
                "value": "no"
            },
            "alignment_loss": {
                "desc": None,
                "value": False
            },
            "n_warmup_steps": {
                "desc": None,
                "value": 10000
            },
            "schedule_every": {
                "desc": None,
                "value": 400
            },
            "uniformity_loss": {
                "desc": None,
                "value": False
            },
            "cifar_batch_size": {
                "desc": None,
                "value": 128
            },
            "do_checkpointing": {
                "desc": None,
                "value": True
            },
            "mismatched_pairs": {
                "desc": None,
                "value": False
            },
            "n_embeds_to_save": {
                "desc": None,
                "value": 512
            },
            "use_train_as_val": {
                "desc": None,
                "value": False
            },
            "cosine_align_loss": {
                "desc": None,
                "value": False
            },
            "encoder1_modality": {
                "desc": None,
                "value": "image"
            },
            "encoder2_modality": {
                "desc": None,
                "value": "text"
            },
            "openai_clip_model": {
                "desc": None,
                "value": "ViT-B/32"
            },
            "scaled_denominator": {
                "desc": None,
                "value": False
            },
            "train_from_scratch": {
                "desc": None,
                "value": False
            },
            "clip_projection_dim": {
                "desc": None,
                "value": 128
            },
            "intra_modality_loss": {
                "desc": None,
                "value": False
            },
            "selected_clip_model": {
                "desc": None,
                "value": "clip_finetuned_temp"
            },
            "uniform_cyclic_loss": {
                "desc": None,
                "value": False
            },
            "train_only_one_batch": {
                "desc": None,
                "value": False
            },
            "use_cached_val_batch": {
                "desc": None,
                "value": True
            },
            "visualize_embeddings": {
                "desc": None,
                "value": False
            },
            "cross_uniformity_loss": {
                "desc": None,
                "value": False
            },
            "cyclic_direction_loss": {
                "desc": None,
                "value": False
            },
            "grad_cache_multiplier": {
                "desc": None,
                "value": 16
            },
            "learnable_temperature": {
                "desc": None,
                "value": False
            },
            "second_caption_offset": {
                "desc": None,
                "value": False
            },
            "show_incorrect_images": {
                "desc": None,
                "value": False
            },
            "train_from_pretrained": {
                "desc": None,
                "value": True
            },
            "use_small_trainloader": {
                "desc": None,
                "value": False
            },
            "validation_batch_size": {
                "desc": None,
                "value": 512
            },
            "cosine_uniformity_loss": {
                "desc": None,
                "value": False
            },
            "delete_val_batch_first": {
                "desc": None,
                "value": False
            },
            "finetune_clip_backbone": {
                "desc": None,
                "value": True
            },
            "common_projection_layer": {
                "desc": None,
                "value": False
            },
            "loss_file_name_template": {
                "desc": None,
                "value": "Ttemp_loss_seed_trainmode_captionencoder_dim_val_bsize_dataset_vmodel_pretrained_EVAL"
            },
            "remove_contrastive_loss": {
                "desc": None,
                "value": False
            },
            "validation_dataset_size": {
                "desc": None,
                "value": 512
            },
            "continue_from_checkpoint": {
                "desc": None,
                "value": False
            },
            "shared_transformer_layers": {
                "desc": None,
                "value": False
            },
            "zero_shot_acc_num_workers": {
                "desc": None,
                "value": 4
            },
            "intra_modality_temperature": {
                "desc": None,
                "value": 0.01
            },
            "save_encoder_hidden_states": {
                "desc": None,
                "value": False
            },
            "finetune_multi_layer_projection": {
                "desc": None,
                "value": False
            }
}
    },
    {
        'summary': {
        "image_S7": 0.4948714077472687,
        "text_variance1": 0.26730602979660034,
        "image_variance1": 0.25609931349754333,
        "std_dev_linear_probe_accuracy": 0,
        "text_rank": 72,
        "train_uniformity_loss": -3.7688064575195312,
        "linear_seperability_accuracy": 0.6926829268292682,
        "mean_text_text_cosine_similarity": 0.005351090803742409,
        "average_intra_modality_cosine_similarity": 0.004396102158352733,
        "image_rank": 76,
        "centroid_cosine_similarity": 0.01491331309080124,
        "cifar10_inter_modality_loss": 1.9769597053527832,
        "cifar10_mean_cosine_similarity": 0.20437726378440857,
        "non_similar_mean_cosine_similarity": -0.0011912111658602953,
        "avg_S": 1.6214076280593872,
        "image_S4": 1.0681287050247192,
        "uniformity": -3.7688064575195312,
        "full_image_rank": 128,
        "full_text_rank": 128,
        "image_variance3": 0.06250832974910736,
        "image_intermodality_rsa": 0.676363280454817,
        "train_intermodality_loss": 2.9084951877593994,
        "train_uniform_cyclic_loss": 0.057480406016111374,
        "val_image_retrieval_accuracy": 0.6484375,
        "cifar10_image_uniformity_loss": -2.600628614425659,
        "cifar10_temp_scaled_inter_modality_loss": 3.2347235679626465,
        "first_lt1_value": 0.99840646982193,
        "image_variance0": 0.475539892911911,
        "image_variance2": 0.1279510259628296,
        "train_cyclic_dir_loss": 0.010536853224039078,
        "train_cross_uniformity_loss": -3.816915273666382,
        "cifar10_linear_probe_accuracy": -1,
        "pearson_image_intermodality_rsa": 0.7719811246341673,
        "text_S3": 1.3955097198486328,
        "image_S6": 0.6511578559875488,
        "text_variance3": 0.061433907598257065,
        "text_variance4": 0.03195323422551155,
        "image_variance4": 0.03557759150862694,
        "cifar100_linear_probe_accuracy": -1,
        "_step": 20800,
        "text_variance2": 0.13228817284107208,
        "image_S0": 3.877672910690307,
        "image_variance5": 0.02133735455572605,
        "text_intermodality_rsa": 0.6714822050302479,
        "cifar10_centroid_euclidean_distance": 0.5331392884254456,
        "svd": -100,
        "text_S1": 2.9131617546081543,
        "image_S5": 0.8255530595779419,
        "mean_cosine_similarity": 0.6565735340118408,
        "centroid_euclidean_distance": 0.11177795380353928,
        "text_S0": 3.876542329788208,
        "text_S5": 0.7560135126113892,
        "image_S1": 2.847761631011963,
        "image_S2": 2.0096843242645264,
        "image_variance7": 0.00769825978204608,
        "train_cyclic_loss": 3.5369542968750034,
        "train_intramodality_loss": -100,
        "text_variance7": 0.005312326364219189,
        "train_rsa_loss": -100,
        "train_pearson_loss": -100,
        "rsa_before_interchanging": 0.4619179526562026,
        "pearson_text_intermodality_rsa": 0.7707315405722833,
        "caltech101_linear_probe_accuracy": -1,
        "val_image_classification_accuracy": 0.666015625,
        "_timestamp": 1716210743.8182118,
        "pearson_rsa_before_interchanging": 0.6140709863831426,
        "cifar10_val_image_classification_accuracy": 0.7461,
        "mean_pairwise_euclidean_distance": 0.819271445274353,
        "text_S2": 2.0408215522766113,
        "text_S4": 1.0084002017974854,
        "text_S6": 0.5706405639648438,
        "temperature": 0.010000000000000004,
        "image_variance6": 0.01328822411596775,
        "train_alignment_loss": 0.6868529319763184,
        "dtd_linear_probe_accuracy": -1,
        "_wandb.runtime": 12475,
        "_runtime": 12445.936551809313,
        "average_linear_probe_accuracy": -1,
        "text_S7": 0.4101710319519043,
        "image_S3": 1.4123473167419434,
        "text_variance0": 0.47360682487487793,
        "text_variance5": 0.01792655698955059,
        "text_variance6": 0.010172848589718342,
        "train_total_loss": -3.9903736114501953,
        "mean_image_image_cosine_similarity": 0.003441113512963057
        },
        'config': {
            "lr": {
                "desc": None,
                "value": 0.000001
            },
            "host": {
                "desc": None,
                "value": "cirrus"
            },
            "seed": {
                "desc": None,
                "value": 42
            },
            "_wandb": {
                "desc": None,
                "value": {
                "t": {
                    "1": [
                    1,
                    5,
                    11,
                    41,
                    49,
                    51,
                    53,
                    55,
                    100
                    ],
                    "2": [
                    1,
                    5,
                    11,
                    41,
                    49,
                    51,
                    53,
                    55,
                    100
                    ],
                    "3": [
                    23,
                    37
                    ],
                    "4": "3.10.13",
                    "5": "0.16.0",
                    "6": "4.34.0",
                    "8": [
                    5
                    ],
                    "13": "linux-x86_64"
                },
                "framework": "huggingface",
                "start_time": 1716198297.88166,
                "cli_version": "0.16.0",
                "is_jupyter_run": False,
                "python_version": "3.10.13",
                "is_kaggle_kernel": False,
                "huggingface_version": "4.34.0"
                }
            },
            "dataset": {
                "desc": None,
                "value": "mscoco"
            },
            "csv_path": {
                "desc": None,
                "value": "stats/"
            },
            "n_epochs": {
                "desc": None,
                "value": 20
            },
            "rsa_loss": {
                "desc": None,
                "value": False
            },
            "svd_loss": {
                "desc": None,
                "value": False
            },
            "max_steps": {
                "desc": None,
                "value": None
            },
            "batch_size": {
                "desc": None,
                "value": 64
            },
            "grad_cache": {
                "desc": None,
                "value": False
            },
            "save_every": {
                "desc": None,
                "value": 200
            },
            "W_layer_gap": {
                "desc": None,
                "value": -1
            },
            "cifar10_acc": {
                "desc": None,
                "value": True
            },
            "cuda_device": {
                "desc": None,
                "value": "cuda:1"
            },
            "cyclip_loss": {
                "desc": None,
                "value": False
            },
            "num_workers": {
                "desc": None,
                "value": 12
            },
            "one_encoder": {
                "desc": None,
                "value": False
            },
            "same_inputs": {
                "desc": None,
                "value": False
            },
            "save_losses": {
                "desc": None,
                "value": False
            },
            "simclr_loss": {
                "desc": None,
                "value": False
            },
            "temperature": {
                "desc": None,
                "value": 0.01
            },
            "loss_weights": {
                "desc": None,
                "value": {
                "image_to_text_weight": 0.5,
                "text_to_image_weight": 0.5
                }
            },
            "pearson_loss": {
                "desc": None,
                "value": False
            },
            "same_encoder": {
                "desc": None,
                "value": False
            },
            "vision_model": {
                "desc": None,
                "value": "VIT"
            },
            "weight_decay": {
                "desc": None,
                "value": 0.1
            },
            "hf_clip_model": {
                "desc": None,
                "value": "openai/clip-vit-base-patch32"
            },
            "use_scheduler": {
                "desc": None,
                "value": "no"
            },
            "alignment_loss": {
                "desc": None,
                "value": True
            },
            "n_warmup_steps": {
                "desc": None,
                "value": 10000
            },
            "schedule_every": {
                "desc": None,
                "value": 400
            },
            "uniformity_loss": {
                "desc": None,
                "value": True
            },
            "cifar_batch_size": {
                "desc": None,
                "value": 128
            },
            "do_checkpointing": {
                "desc": None,
                "value": True
            },
            "mismatched_pairs": {
                "desc": None,
                "value": False
            },
            "n_embeds_to_save": {
                "desc": None,
                "value": 512
            },
            "use_train_as_val": {
                "desc": None,
                "value": False
            },
            "cosine_align_loss": {
                "desc": None,
                "value": False
            },
            "encoder1_modality": {
                "desc": None,
                "value": "image"
            },
            "encoder2_modality": {
                "desc": None,
                "value": "text"
            },
            "openai_clip_model": {
                "desc": None,
                "value": "ViT-B/32"
            },
            "scaled_denominator": {
                "desc": None,
                "value": False
            },
            "train_from_scratch": {
                "desc": None,
                "value": False
            },
            "clip_projection_dim": {
                "desc": None,
                "value": 128
            },
            "intra_modality_loss": {
                "desc": None,
                "value": False
            },
            "selected_clip_model": {
                "desc": None,
                "value": "clip_finetuned_temp"
            },
            "uniform_cyclic_loss": {
                "desc": None,
                "value": False
            },
            "train_only_one_batch": {
                "desc": None,
                "value": False
            },
            "use_cached_val_batch": {
                "desc": None,
                "value": True
            },
            "visualize_embeddings": {
                "desc": None,
                "value": False
            },
            "cross_uniformity_loss": {
                "desc": None,
                "value": True
            },
            "cyclic_direction_loss": {
                "desc": None,
                "value": False
            },
            "grad_cache_multiplier": {
                "desc": None,
                "value": 16
            },
            "learnable_temperature": {
                "desc": None,
                "value": False
            },
            "second_caption_offset": {
                "desc": None,
                "value": False
            },
            "show_incorrect_images": {
                "desc": None,
                "value": False
            },
            "train_from_pretrained": {
                "desc": None,
                "value": True
            },
            "use_small_trainloader": {
                "desc": None,
                "value": False
            },
            "validation_batch_size": {
                "desc": None,
                "value": 512
            },
            "cosine_uniformity_loss": {
                "desc": None,
                "value": False
            },
            "delete_val_batch_first": {
                "desc": None,
                "value": False
            },
            "finetune_clip_backbone": {
                "desc": None,
                "value": True
            },
            "common_projection_layer": {
                "desc": None,
                "value": False
            },
            "loss_file_name_template": {
                "desc": None,
                "value": "Ttemp_loss_seed_trainmode_captionencoder_dim_val_bsize_dataset_vmodel_pretrained_EVAL"
            },
            "remove_contrastive_loss": {
                "desc": None,
                "value": False
            },
            "validation_dataset_size": {
                "desc": None,
                "value": 512
            },
            "continue_from_checkpoint": {
                "desc": None,
                "value": False
            },
            "shared_transformer_layers": {
                "desc": None,
                "value": False
            },
            "zero_shot_acc_num_workers": {
                "desc": None,
                "value": 4
            },
            "intra_modality_temperature": {
                "desc": None,
                "value": 0.01
            },
            "save_encoder_hidden_states": {
                "desc": None,
                "value": False
            },
            "finetune_multi_layer_projection": {
                "desc": None,
                "value": False
            }
        }
    }
]