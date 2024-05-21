from matplotlib import pyplot as plt


import numpy as np


default_clip_summary = {
  "text_variance5": 0.02687590569257736,
  "image_variance2": 0.13273699581623075,
  "image_variance5": 0.0282245259732008,
  "text_intermodality_rsa": 0.7960489110189808,
  "train_cross_uniformity_loss": -1.7636101245880127,
  "svd": -100,
  "image_S6": 0.6734693050384521,
  "cifar10_centroid_euclidean_distance": 0.3599262833595276,
  "train_rsa_loss": -100,
  "cifar100_linear_probe_accuracy": -1,
  "image_variance6": 0.016566487029194832,
  "train_alignment_loss": 0.2892352044582367,
  "image_intermodality_rsa": 0.7810457093722973,
  "rsa_before_interchanging": 0.627238992656837,
  "val_image_retrieval_accuracy": 0.6875,
  "average_intra_modality_cosine_similarity": 0.5724155604839325,
  "text_S0": 5.372421741485596,
  "temperature": 0.010000000000000004,
  "text_variance0": 0.46951574087142944,
  "mean_cosine_similarity": 0.8553823828697205,
  "dtd_linear_probe_accuracy": -1,
  "cifar10_inter_modality_loss": 2.2046449184417725,
  "pearson_text_intermodality_rsa": 0.8186489053136927,
  "_wandb.runtime": 8783,
  "text_S1": 2.532754421234131,
  "train_intermodality_loss": 1.289703607559204,
  "image_S1": 2.454017639160156,
  "_timestamp": 1716179887.8613143,
  "image_S7": 0.5066552758216858,
  "text_variance3": 0.08123276382684708,
  "image_variance7": 0.009569224901497364,
  "std_dev_linear_probe_accuracy": 0,
  "caltech101_linear_probe_accuracy": -1,
  "non_similar_mean_cosine_similarity": 0.5415878295898438,
  "text_S4": 1.2125844955444336,
  "text_S5": 0.8967400789260864,
  "cifar10_temp_scaled_inter_modality_loss": 0.7960696220397949,
  "image_S5": 0.8878350257873535,
  "uniformity": -1.6356980800628662,
  "pearson_rsa_before_interchanging": 0.6684574507400891,
  "image_S2": 1.9166300296783447,
  "image_S3": 1.466128706932068,
  "mean_text_text_cosine_similarity": 0.5556365251541138,
  "text_rank": 41,
  "first_lt1_value": 0.959197461605072,
  "image_variance0": 0.46854427456855774,
  "pearson_image_intermodality_rsa": 0.8069300909573313,
  "_step": 11000,
  "image_S0": 5.298120498657227,
  "text_S2": 2.0020127296447754,
  "train_cyclic_dir_loss": 0.007097981404513121,
  "image_variance4": 0.04968678578734398,
  "average_linear_probe_accuracy": -1,
  "cifar10_image_uniformity_loss": -0.8831652998924255,
  "image_rank": 41,
  "text_variance6": 0.016010865569114685,
  "full_text_rank": 64,
  "text_variance7": 0.008439578115940094,
  "train_total_loss": 1.289703607559204,
  "linear_seperability_accuracy": 0.996341463414634,
  "cifar10_linear_probe_accuracy": -1,
  "text_S3": 1.5612773895263672,
  "text_S6": 0.6774754524230957,
  "text_variance4": 0.04898339882493019,
  "full_image_rank": 64,
  "train_cyclic_loss": 1.6099858398437514,
  "train_uniformity_loss": -1.6356980800628662,
  "cifar10_mean_cosine_similarity": 0.7608008980751038,
  "avg_S": 1.843231439590454,
  "image_S4": 1.1715130805969238,
  "train_pearson_loss": -100,
  "mean_pairwise_euclidean_distance": 0.5337882041931152,
  "mean_image_image_cosine_similarity": 0.5891945958137512,
  "text_S7": 0.4905846416950226,
  "image_variance1": 0.21748754382133484,
  "image_variance3": 0.07718412578105927,
  "train_intramodality_loss": -100,
  "centroid_cosine_similarity": 0.9462392330169678,
  "cifar10_val_image_classification_accuracy": 0.779,
  "text_variance1": 0.21455679833889008,
  "text_variance2": 0.13438494503498075,
  "centroid_euclidean_distance": 0.249198317527771,
  "val_image_classification_accuracy": 0.671875,
  "_runtime": 8748.596397638321,
  "train_uniform_cyclic_loss": 0.0283429604023695
}



clip_a_u_summary = {
  "linear_seperability_accuracy": 0.5073170731707317,
  "cifar10_centroid_euclidean_distance": 0.45679759979248047,
  "cifar10_temp_scaled_inter_modality_loss": 2.5020925998687744,
  "text_S5": 1.3101298809051514,
  "image_S5": 1.3351781368255615,
  "image_variance6": 0.01532598864287138,
  "train_alignment_loss": 0.40488266944885254,
  "cifar10_inter_modality_loss": 1.9039623737335205,
  "text_S0": 4.965303897857666,
  "text_rank": 51,
  "image_variance2": 0.16313596069812775,
  "full_image_rank": 64,
  "train_uniform_cyclic_loss": 0.06678057461977005,
  "non_similar_mean_cosine_similarity": 0.0002674572169780731,
  "mean_pairwise_euclidean_distance": 0.6150933504104614,
  "train_cyclic_dir_loss": 0.012596556916832924,
  "train_intramodality_loss": -100,
  "caltech101_linear_probe_accuracy": -1,
  "text_variance1": 0.246175080537796,
  "text_S1": 3.964444160461426,
  "full_text_rank": 64,
  "text_variance0": 0.3873473107814789,
  "text_S2": 3.2675223350524902,
  "train_intermodality_loss": 3.0726025104522705,
  "cifar10_image_uniformity_loss": -2.273869037628174,
  "svd": -100,
  "uniformity": -3.63630485534668,
  "image_intermodality_rsa": 0.7844493151994723,
  "centroid_cosine_similarity": 0.6897773742675781,
  "image_rank": 51,
  "text_variance4": 0.05197184532880783,
  "text_variance7": 0.007803826592862606,
  "train_total_loss": -0.15881967544555664,
  "val_image_retrieval_accuracy": 0.6484375,
  "cifar10_mean_cosine_similarity": 0.2739006578922272,
  "_step": 11000,
  "text_S4": 1.817919135093689,
  "image_variance4": 0.05147012695670128,
  "_wandb.runtime": 8784,
  "image_S0": 4.970821380615234,
  "average_intra_modality_cosine_similarity": 0.0007694761516177095,
  "text_variance2": 0.16737449169158936,
  "text_variance6": 0.014127553440630436,
  "image_variance0": 0.3875432908535004,
  "image_variance3": 0.09551554173231123,
  "train_cross_uniformity_loss": -3.647756099700928,
  "image_S7": 0.7398295402526855,
  "_timestamp": 1716179969.9280572,
  "temperature": 0.010000000000000004,
  "cifar10_linear_probe_accuracy": -1,
  "cifar100_linear_probe_accuracy": -1,
  "cifar10_val_image_classification_accuracy": 0.7981,
  "train_uniformity_loss": -3.63630485534668,
  "rsa_before_interchanging": 0.6263057537336604,
  "average_linear_probe_accuracy": -1,
  "mean_text_text_cosine_similarity": 0.00013043924991507083,
  "pearson_rsa_before_interchanging": 0.723994982383356,
  "text_S3": 2.4953460693359375,
  "text_variance5": 0.027208909392356873,
  "image_variance5": 0.028152653947472572,
  "val_image_classification_accuracy": 0.638671875,
  "train_pearson_loss": -100,
  "text_intermodality_rsa": 0.7898580255779967,
  "pearson_image_intermodality_rsa": 0.8429207218705499,
  "_runtime": 8750.529324293137,
  "image_variance7": 0.008626212365925312,
  "train_cyclic_loss": 4.261853515625004,
  "mean_cosine_similarity": 0.7975587248802185,
  "image_S1": 3.992382526397705,
  "image_S6": 0.9889044761657716,
  "text_variance3": 0.0979910045862198,
  "std_dev_linear_probe_accuracy": 0,
  "pearson_text_intermodality_rsa": 0.8490446254094055,
  "text_S6": 0.9476362466812134,
  "image_variance1": 0.25023025274276733,
  "centroid_euclidean_distance": 0.04234221577644348,
  "image_S3": 2.46236252784729,
  "image_S4": 1.8081984519958496,
  "train_rsa_loss": -100,
  "first_lt1_value": 0.9567292332649232,
  "dtd_linear_probe_accuracy": -1,
  "avg_S": 2.4338324069976807,
  "text_S7": 0.7023578882217407,
  "image_S2": 3.2211320400238037,
  "mean_image_image_cosine_similarity": 0.0014085130533203485
}


from run_summaries import runs

names = ['L_{CLIP}', 'L_{CLIP+Align+Uniform+XUniform}']
colors = ['tab:blue', 'tab:red', 'tab:green']
# plot bar chart of average of image and text variances

# summaries = [default_clip_summary, clip_a_u_summary]

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
def plot_cumulative_explained_vars(runs):
        
    width=0.45
  
  
    all_average_variances = []
  
  
    for i, run in enumerate(runs):
          
          avg_vars = []
  
          for j in range(8):
              
              average_variance = (run['summary'][f'image_variance{j}'] + run['summary'][f'text_variance{j}']) / 2
              avg_vars.append(average_variance)
  
          all_average_variances.append(avg_vars)
  
    for i, run in enumerate(runs):
        
        # name = summary['name']
        name = names[i]
        color = colors[i]
        plt.plot(np.arange(start=1, stop=64, step=8), np.cumsum(all_average_variances[i]), label=r'${}$'.format(name), color=f'{color}')
  
      
        
    # plt.xticks(np.arange(8) + width / 2, ('0-7', '8-15', '16-23', '24-31', '32-39', '40-47', '48-55', '56-63'))
  
    plt.legend(fontsize="22")
  
    plt.xlabel('Dimensions')
  
    plt.ylabel('PCA Explained variance')
  
    plt.show()





def plot_dim_acc_graph(runs, fields=['cifar10_val_image_classification_accuracy', 'cifar100_val_image_classification_accuracy'], group_by='uniformity_loss', xlabel='CLIP Dimensionality', ylabel='CIFAR10 Validation Accuracy', group_labels=['L_{CLIP}', 'L_{CLIP+Align+Uniform+XUniform}'], markers=['o', 'x'], colors=colors):
    

    collected_metric_values = []

    collected_dims = []

    collected_groups = []

    
    # gather data
    for i, run in enumerate(runs):
        
        collected_dims.append(run['config']['clip_projection_dim']['value'])
        # collected_metric_values.append(run['summary'][field])
        collected_groups.append(run['config'][group_by]['value'])

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
                
                if run['config']['clip_projection_dim']['value'] == dim and run['config'][group_by]['value'] == group:
                    
                    avg_metric = 0
                    
                    for field in fields:
                      avg_metric += run['summary'][field]

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
    plt.show()


    
    pass



# plot_explained_vars(summaries)

# print(runs[-2:])

plot_cumulative_explained_vars(runs[-2:])

# plot_dim_acc_graph(runs, fields=['cifar10_val_image_classification_accuracy', 'cifar100_val_image_classification_accuracy'], group_by='uniformity_loss', xlabel='CLIP Dimensionality', ylabel='Average Zero-Shot Accuracy', group_labels=['L_{CLIP}', 'L_{CLIP+Align+Uniform}'], markers=['o', 'x'], colors=['tab:blue', 'tab:red'])
# plot_dim_acc_graph(runs, fields=['cifar10_linear_probe_accuracy', 'cifar100_linear_probe_accuracy'], group_by='uniformity_loss', xlabel='CLIP Dimensionality', ylabel='Average Linear Probe Accuracy', group_labels=['L_{CLIP}', 'L_{CLIP+Align+Uniform}'], markers=['o', 'x'], colors=['tab:blue', 'tab:red'])
# plot_dim_acc_graph(runs, field='val_image_classification_accuracy', group_by='uniformity_loss', xlabel='CLIP Dimensionality', ylabel='I -> T', group_labels=['L_{CLIP}', 'L_{CLIP+Align+Uniform}'], markers=['o', 'x'], colors=['tab:blue', 'tab:red'])