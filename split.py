from cfg import *
import numpy as np
import nibabel as nib

for i, fname in enumerate(fnames_data):
	print("Subject " + subs[i])
	img = nib.load(fname)
	x = img.get_fdata()
	train_img = nib.Nifti1Image(x[:, :, :,:946], img.affine)
	test_img = nib.Nifti1Image(x[:, :, :,946:], img.affine)
	nib.save(train_img, train_d + subs[i] + "_train.nii.gz")
	nib.save(test_img, test_d + subs[i] + "_test.nii.gz")
