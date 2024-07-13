import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from nilearn import plotting
import imageio
import os



data_dir = 'C:/Users/lusim/OneDrive/Desktop/neurocomp/study-forrest/ds000113'
fnames_data = [ f'{data_dir}/sub-01/ses-movie/func/sub-01_ses-movie_task-movie_run-{i}_bold.nii.gz' for i in range(1,9)]

fmri_imgs = []
fmri_data = []

for f in fnames_data:
    img = nib.load(f)
    data = img.get_fdata()
    
    fmri_imgs.append(img)
    fmri_data.append(data)
    

data = np.concatenate((fmri_data), axis = 3)
affine = nib.load(fnames_data[0]).affine
img = nib.Nifti1Image(data, affine) # Create a new NiBabel image
print('Data correctly loaded')

# Parameters for the GIF creation
slice_index = data.shape[2] // 2  # Middle slice
timepoints = data.shape[3]        # Number of timepoints

# Create a list to store the filenames of the generated images
filenames = []

# Generate images for each timepoint
for time_point in range(timepoints):
    slice_data = data[:, :, slice_index, time_point]
    volume_img = nib.Nifti1Image(data[..., time_point], img.affine)
    
    # Plot the image
    fig, ax = plt.subplots()
    display = plotting.plot_epi(volume_img, colorbar=True, cbar_tick_format="%i", axes=ax)
    plt.title(f'Slice {slice_index}, Time Frame {time_point}')
    
    # Save the figure
    filename = f'temp_{time_point}.png'
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()



filenames = sorted([f'temp_{i}.png' for i in range(1793)])
output_gif = 'fmri_midslice_animation.gif'
duration = 0.5

# Create the GIF
with imageio.get_writer(output_gif, mode='I', duration=duration) as writer:
    for filename in filenames:
        image = imageio.v2.imread(filename)
        writer.append_data(image)

print(f"GIF saved as {output_gif}")

# Remove temporary files
for filename in filenames:
    os.remove(filename)
