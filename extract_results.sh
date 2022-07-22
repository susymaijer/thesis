#!/bin/bash

echo "Get all the results for a specific nnUnet configuration."
echo "This means; OUTPUT/task/config, RESULTS_FOLDER[cv_niftis_postprocessed and fold_X]"
echo ""

read -p "Enter task:" task

read -p "Enter config [3d_lowres, 3d_cascade_fullres, 3d_fullres]:" config

read -p "Enter trainer [UNETR,UNETRLarge,Hybrid,Hybrid2,Hybrid2LR,empty]:" trainer

# lowres of fullres
if [ $config != "3d_cascade_fullres" ];
then
   if [ ! -z $trainer ];
   then
      trainer="nnUNetTrainerV2_$trainer"
   fi
   if [ -z $trainer ];
   then
      trainer="nnUNetTrainerV2"
   fi
fi
# cascade
if [ $config == "3d_cascade_fullres" ];
then
   if [ ! -z $trainer ];
   then
      trainer="nnUNetTrainerV2CascadeFullRes_$trainer"
   fi
   if [ -z $trainer ];
   then
      trainer="nnUNetTrainerV2CascadeFullRes"
   fi
fi

download="$DOWNLOAD/$task/$config/$trainer"
mkdir -p $download

# Extract the final segmentations of training and testing dataset 
cd $OUTPUT/$task/$config/$trainer
ls -l
zip -r inference.zip *
mv inference.zip $download

# Extract the cross validation ensembled results of all the training images
cd $RESULTS_FOLDER/nnUNet/$config/Task$task/${trainer}__nnUNetPlansv2.1/cv_niftis_postprocessed
zip -r cv_niftis_postprocessed.zip *
mv cv_niftis_postprocessed.zip $download

# Extract all the fold specific results 
extract_fold(){

	# variables so we don't type wrong
	fold="fold_$1"
	fold_dir="${fold}_results"
	fold_dir_zip="${fold}_dir.zip"
	
	# we are either in a fold or in cv_niftis_postprocessed. so go one folder up
	cd ..
	cd $fold
	echo $pwd
	mkdir -p $fold_dir
	
	# get all the relevant files
	cp debug.json $fold_dir
	cp p* $fold_dir -r
	cp validation_* $fold_dir -r
	
	# zip all the files and move
	cd $fold_dir
	zip -r $fold_dir_zip *
        mv $fold_dir_zip $download
	cd ..
}
for i in 0 1 2 3 4
do
	echo "Extracting results for fold $i"
	extract_fold $i
done

