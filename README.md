# Study Forrest

1. To download the StudyForrest fMRI files (first run for each subject), run the following commands:

``` bash
mkdir study-forrest
cd study-forrest
datalad install https://github.com/OpenNeuroDatasets/ds000113
cd ds000113
datalad get sub-*/ses-movie/func/*_ses-movie_task-movie_run-1_bold.nii.gz
```

2. Install all dependencies included in the `Pipfile`.