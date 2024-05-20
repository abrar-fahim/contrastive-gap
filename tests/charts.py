from matplotlib import pyplot as plt


import numpy as np


default_clip_summary = {
  'name': 'L_{CLIP}',
  'color': 'tab:blue',
  "dtd_linear_probe_accuracy": -1,
  "average_linear_probe_accuracy": -1,
  "pearson_text_intermodality_rsa": 0.8012633330669773,
  "mean_text_text_cosine_similarity": 0.43775272369384766,
  "text_S7": 0.6130275726318359,
  "_timestamp": 1715854845.732164,
  "first_lt1_value": 0.9530285596847534,
  "cifar10_image_uniformity_loss": -1.1080471277236938,
  "image_S4": 1.3223600387573242,
  "image_rank": 46,
  "text_variance2": 0.12417963147163393,
  "full_image_rank": 64,
  "svd": -100,
  "text_S2": 2.174670457839966,
  "text_S3": 1.6682332754135132,
  "_runtime": 17822.480667114258,
  "std_dev_linear_probe_accuracy": 0,
  "pearson_image_intermodality_rsa": 0.7982718977224249,
  "val_image_classification_accuracy": 0.5675146579742432,
  "image_variance5": 0.03059515729546547,
  "train_alignment_loss": -100,
  "train_intermodality_loss": 1.9684678316116333,
  "train_uniform_cyclic_loss": -100,
  "pearson_rsa_before_interchanging": 0.656863936490344,
  "average_intra_modality_cosine_similarity": 0.4534625858068466,
  "image_S1": 2.7181396484375,
  "text_variance6": 0.018166236579418182,
  "centroid_cosine_similarity": 0.93654865026474,
  "cifar10_inter_modality_loss": 2.1818180084228516,
  "train_cross_uniformity_loss": -100,
  "cifar10_linear_probe_accuracy": -1,
  "cifar10_temp_scaled_inter_modality_loss": 1.0003360509872437,
  "_wandb.runtime": 17904,
  "text_variance4": 0.04713404178619385,
  "image_variance7": 0.012269251979887486,
  "_step": 32200,
  "image_variance2": 0.12256178259849548,
  "linear_seperability_accuracy": 0.9853300733496332,
  "mean_pairwise_euclidean_distance": 0.625549852848053,
  "image_S3": 1.658820867538452,
  "image_variance0": 0.48373785614967346,
  "train_total_loss": 1.9684678316116333,
  "cifar10_mean_cosine_similarity": 0.6767077445983887,
  "mean_image_image_cosine_similarity": 0.4691724479198456,
  "text_S5": 1.0292742252349854,
  "text_variance1": 0.21411898732185364,
  "train_intramodality_loss": -100,
  "centroid_euclidean_distance": 0.24122202396392825,
  "cifar10_val_image_classification_accuracy": 0.7644,
  "text_variance7": 0.010327043943107128,
  "train_pearson_loss": -100,
  "image_intermodality_rsa": 0.7766489905358942,
  "cifar10_centroid_euclidean_distance": 0.4226909875869751,
  "rsa_before_interchanging": 0.6216565460804845,
  "non_similar_mean_cosine_similarity": 0.42470020055770874,
  "avg_S": 2.0061702728271484,
  "text_S0": 5.564552307128906,
  "text_rank": 45,
  "image_variance3": 0.07758976519107819,
  "image_S2": 2.0960769653320312,
  "image_S7": 0.6577264070510864,
  "temperature": 0.010000000000000004,
  "text_variance5": 0.02885878086090088,
  "train_cyclic_loss": -100,
  "caltech101_linear_probe_accuracy": -1,
  "text_S6": 0.8130776286125183,
  "image_S5": 1.0379798412322998,
  "text_variance0": 0.4835651516914368,
  "text_variance3": 0.07365016639232635,
  "text_intermodality_rsa": 0.7804859831452867,
  "uniformity": -2.063627004623413,
  "train_rsa_loss": -100,
  "image_variance6": 0.019877487793564796,
  "mean_cosine_similarity": 0.8017466068267822,
  "image_variance1": 0.20407214760780337,
  "image_variance4": 0.04929658770561218,
  "cifar100_linear_probe_accuracy": -1,
  "text_S4": 1.327985763549805,
  "image_S0": 5.538852214813232,
  "image_S6": 0.8358714580535889,
  "full_text_rank": 64,
  "text_S1": 2.8585410118103027,
  "train_uniformity_loss": -2.063627004623413,
  "val_image_retrieval_accuracy": 0.5772994160652161
}



clip_a_u_x_summary = {
  'name': 'L_{CLIP+Align+Uniform+XUniform}',
  'color': 'tab:red',
  "val_image_retrieval_accuracy": 0.5831702351570129,
  "cifar10_linear_probe_accuracy": -1,
  "mean_pairwise_euclidean_distance": 0.726881206035614,
  "svd": -100,
  "image_variance2": 0.1629336029291153,
  "image_variance6": 0.043716054409742355,
  "pearson_image_intermodality_rsa": 0.7425755225403292,
  "text_S2": 3.2224063873291016,
  "text_variance0": 0.25864851474761963,
  "train_alignment_loss": 0.5574826598167419,
  "cifar10_image_uniformity_loss": -2.669571876525879,
  "full_image_rank": 64,
  "_wandb.runtime": 26788,
  "text_S1": 3.564453601837158,
  "image_S3": 2.9399971961975098,
  "image_S7": 1.1405243873596191,
  "text_variance7": 0.018994230777025223,
  "train_rsa_loss": -100,
  "train_total_loss": -2.5223867893218994,
  "mean_cosine_similarity": 0.7212586998939514,
  "pearson_rsa_before_interchanging": 0.5595336109344499,
  "text_intermodality_rsa": 0.6386428426714488,
  "non_similar_mean_cosine_similarity": -0.00030936949769966304,
  "text_S0": 4.058292388916016,
  "text_S7": 1.093752384185791,
  "image_S1": 3.5674800872802734,
  "temperature": 0.010000000000000004,
  "train_uniformity_loss": -3.765352725982666,
  "_timestamp": 1715864149.4026656,
  "text_variance1": 0.19925382733345032,
  "text_variance6": 0.041671570390462875,
  "text_S3": 2.9442739486694336,
  "text_S6": 1.6221293210983276,
  "image_S2": 3.222433567047119,
  "image_S5": 2.2569162845611572,
  "text_rank": 62,
  "image_intermodality_rsa": 0.6484086644607674,
  "centroid_cosine_similarity": 0.2441727519035339,
  "train_cross_uniformity_loss": -3.7826931476593018,
  "mean_text_text_cosine_similarity": 0.0024080490693449974,
  "mean_image_image_cosine_similarity": 0.002725806785747409,
  "_step": 32200,
  "image_S6": 1.6598825454711914,
  "train_intramodality_loss": -100,
  "dtd_linear_probe_accuracy": -1,
  "pearson_text_intermodality_rsa": 0.7360452494953682,
  "text_S4": 2.6022205352783203,
  "caltech101_linear_probe_accuracy": -1,
  "cifar10_val_image_classification_accuracy": 0.8106,
  "cifar10_temp_scaled_inter_modality_loss": 3.069092035293579,
  "_runtime": 26726.057520627975,
  "image_S0": 3.9905478954315186,
  "image_S4": 2.628680944442749,
  "centroid_euclidean_distance": 0.08265773952007294,
  "std_dev_linear_probe_accuracy": 0,
  "cifar10_mean_cosine_similarity": 0.1702536642551422,
  "full_text_rank": 64,
  "text_variance4": 0.1062462329864502,
  "image_variance0": 0.25018855929374695,
  "image_variance5": 0.0799727588891983,
  "train_pearson_loss": -100,
  "val_image_classification_accuracy": 0.5890411138534546,
  "image_rank": 63,
  "uniformity": -3.765352725982666,
  "image_variance3": 0.13530150055885315,
  "rsa_before_interchanging": 0.4146364323495061,
  "linear_seperability_accuracy": 0.5709046454767727,
  "cifar10_centroid_euclidean_distance": 0.4779911935329437,
  "avg_S": 2.6656694412231445,
  "text_variance3": 0.1357026994228363,
  "image_variance4": 0.10815387964248656,
  "train_cyclic_loss": -100,
  "cifar10_inter_modality_loss": 1.904539465904236,
  "text_variance2": 0.16267742216587067,
  "train_intermodality_loss": 4.468176364898682,
  "average_linear_probe_accuracy": -1,
  "text_S5": 2.2178268432617188,
  "text_variance5": 0.07680553942918777,
  "average_intra_modality_cosine_similarity": 0.002566927927546203,
  "first_lt1_value": 0.9520537853240968,
  "image_variance1": 0.1991196721792221,
  "image_variance7": 0.020613981410861015,
  "train_uniform_cyclic_loss": -100,
  "cifar100_linear_probe_accuracy": -1
}


# plot bar chart of average of image and text variances

summaries = [default_clip_summary, clip_a_u_x_summary]


def plot_explained_vars(summaries):
      

  width=0.35


  all_average_variances = []


  for i, summary in enumerate(summaries):
        

        
        avg_vars = []

        for j in range(8):
            
            average_variance = (summary[f'image_variance{j}'] + summary[f'text_variance{j}']) / 2
            avg_vars.append(average_variance)

        all_average_variances.append(avg_vars)

  for i, summary in enumerate(summaries):
      
      name = summary['name']
      plt.bar(np.arange(8) + i * width, all_average_variances[i], width, label=r'${}$'.format(name), color=f'{summary["color"]}')

    
      
  plt.xticks(np.arange(8) + width / 2, ('0-7', '8-15', '16-23', '24-31', '32-39', '40-47', '48-55', '56-63'))

  plt.legend()

  plt.xlabel('Dimensions')

  plt.ylabel('PCA Explained variance')



  plt.show()




def plot_dim_acc_graph():
    pass

