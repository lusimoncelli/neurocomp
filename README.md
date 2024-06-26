# Study Forrest

1. Install all dependencies included in the `Pipfile`.
2. To download the StudyForrest fMRI files (first run for each subject), run the following commands:

``` bash
mkdir study-forrest
cd study-forrest
datalad install https://github.com/OpenNeuroDatasets/ds000113
cd ds000113
datalad get sub-*/ses-movie/func/*_ses-movie_task-movie_run-1_bold.nii.gz
```

3. Run `cfg.py` and then `split.py`
4. Resample mask size to match the desired size by running `resamplemask.py`
5. Run `functional_space.py` to convert the forrest train and test data form anatomical space to functional space.