#!/bin/bash

cd ./study-forrest/ds000113

for i in {3..20}
do
  formatted_i=$(printf "%02d" $i)
  command="datalad get sub-${formatted_i}/ses-movie/func/sub-${formatted_i}_ses-movie_task-movie_run-*_bold.nii.gz"
  echo $command  # Print the command for verification
  $command  # Execute the command
done
