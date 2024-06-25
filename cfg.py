import os

data_dir = "study-forrest/"
subs = [f'sub-0{i}' for i in range(1,7)]

raw_data_name = data_dir + "ds000113/"
fnames_data = [ f'{raw_data_name}{s}/ses-movie/func/{s}_ses-movie_task-movie_run-1_bold.nii.gz' for s in subs]

# create dirs for train and test data
if not os.path.exists(data_dir + "train/"):
	os.mkdir(data_dir + "train/")
if not os.path.exists(data_dir + "test/"):
	os.mkdir(data_dir + "test/")
if not os.path.exists(data_dir + "functional_space/"):
	os.mkdir(data_dir + "functional_space/")
if not os.path.exists(data_dir+"results/"):
	os.mkdir(data_dir+"results/")

train_d = data_dir + "train/"
test_d = data_dir + "test/"
func_d = data_dir + "functional_space/"
results_dir = data_dir + "results/"

fnames_train = [train_d + s + "_train.nii.gz" for s in subs]
fnames_test = [test_d + s + "_test.nii.gz" for s in subs]

TR_BUFFER_SIZE = 10
DIM_FUNCTIONAL_SPACE = 200
N_FUNCTIONAL_NEIGHBORS = 343
SL_ANAT_RAD = 3