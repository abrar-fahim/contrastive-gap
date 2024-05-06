{
  "_name_or_path": "openai/clip-vit-base-patch32",
  "architectures": [
    "CLIPModel"
  ],
  "initializer_factor": 1.0,
  "logit_scale_init_value": 2.6592,
  "model_type": "clip",
  "projection_dim": 512,
  "text_config": {
    "_name_or_path": "",
    "add_cross_attention": false,
    "architectures": null,
    "attention_dropout": 0.0,
    "bad_words_ids": null,
    "bos_token_id": 0,
    "chunk_size_feed_forward": 0,
    "cross_attention_hidden_size": null,
    "decoder_start_token_id": null,
    "diversity_penalty": 0.0,
    "do_sample": false,
    "dropout": 0.0,
    "early_stopping": false,
    "encoder_no_repeat_ngram_size": 0,
    "eos_token_id": 2,
    "finetuning_task": null,
    "forced_bos_token_id": null,
    "forced_eos_token_id": null,
    "hidden_act": "quick_gelu",
    "hidden_size": 512,
    "id2label": {
      "0": "LABEL_0",
      "1": "LABEL_1"
    },
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 2048,
    "is_decoder": false,
    "is_encoder_decoder": false,
    "label2id": {
      "LABEL_0": 0,
      "LABEL_1": 1
    },
    "layer_norm_eps": 1e-05,
    "length_penalty": 1.0,
    "max_length": 20,
    "max_position_embeddings": 77,
    "min_length": 0,
    "model_type": "clip_text_model",
    "no_repeat_ngram_size": 0,
    "num_attention_heads": 8,
    "num_beam_groups": 1,
    "num_beams": 1,
    "num_hidden_layers": 12,
    "num_return_sequences": 1,
    "output_attentions": false,
    "output_hidden_states": false,
    "output_scores": false,
    "pad_token_id": 1,
    "prefix": null,
    "projection_dim": 512,
    "problem_type": null,
    "pruned_heads": {},
    "remove_invalid_values": false,
    "repetition_penalty": 1.0,
    "return_dict": true,
    "return_dict_in_generate": false,
    "sep_token_id": null,
    "task_specific_params": null,
    "temperature": 1.0,
    "tie_encoder_decoder": false,
    "tie_word_embeddings": true,
    "tokenizer_class": null,
    "top_k": 50,
    "top_p": 1.0,
    "torch_dtype": null,
    "torchscript": false,
    "transformers_version": "4.16.0.dev0",
    "use_bfloat16": false,
    "vocab_size": 49408
  },
  "text_config_dict": null,
  "transformers_version": null,
  "vision_config": {
    "_name_or_path": "",
    "add_cross_attention": false,
    "architectures": null,
    "attention_dropout": 0.0,
    "bad_words_ids": null,
    "bos_token_id": null,
    "chunk_size_feed_forward": 0,
    "cross_attention_hidden_size": null,
    "decoder_start_token_id": null,
    "diversity_penalty": 0.0,
    "do_sample": false,
    "dropout": 0.0,
    "early_stopping": false,
    "encoder_no_repeat_ngram_size": 0,
    "eos_token_id": null,
    "finetuning_task": null,
    "forced_bos_token_id": null,
    "forced_eos_token_id": null,
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "id2label": {
      "0": "LABEL_0",
      "1": "LABEL_1"
    },
    "image_size": 224,
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "is_decoder": false,
    "is_encoder_decoder": false,
    "label2id": {
      "LABEL_0": 0,
      "LABEL_1": 1
    },
    "layer_norm_eps": 1e-05,
    "length_penalty": 1.0,
    "max_length": 20,
    "min_length": 0,
    "model_type": "clip_vision_model",
    "no_repeat_ngram_size": 0,
    "num_attention_heads": 12,
    "num_beam_groups": 1,
    "num_beams": 1,
    "num_hidden_layers": 12,
    "num_return_sequences": 1,
    "output_attentions": false,
    "output_hidden_states": false,
    "output_scores": false,
    "pad_token_id": null,
    "patch_size": 32,
    "prefix": null,
    "projection_dim" : 512,
    "problem_type": null,
    "pruned_heads": {},
    "remove_invalid_values": false,
    "repetition_penalty": 1.0,
    "return_dict": true,
    "return_dict_in_generate": false,
    "sep_token_id": null,
    "task_specific_params": null,
    "temperature": 1.0,
    "tie_encoder_decoder": false,
    "tie_word_embeddings": true,
    "tokenizer_class": null,
    "top_k": 50,
    "top_p": 1.0,
    "torch_dtype": null,
    "torchscript": false,
    "transformers_version": "4.16.0.dev0",
    "use_bfloat16": false
  },
  "vision_config_dict": null
}