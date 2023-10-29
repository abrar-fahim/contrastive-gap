
from training_utils import *

index = 0

# plot pca
# plot_pca_from_file(f'pca_plots/image_coordinates_{index}.npy', f'pca_plots/text_coordinates_{index}.npy')

# plot pca subplots
plot_pca_subplots_from_file('pca_plots/', 0, 20, 4)