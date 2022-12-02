MANUAL STEPS FOR BATCHX

1. Create a new folder batchX within the folder $nnUNet_raw_data_base/p16

2. Upload the zip file in the folder $nnUNet_raw_data_base/p16/batchX

3. Run ./0_createDataSetAndPerformInference.sh 
   --> creates a new nnU-Net task
   --> copies all the existing P16 scans to the new task
   --> converts DICOMs to niftis and adds them to the new nnU-Net task
   --> performs inference on the new scans using the most recent model
   --> LOCATION AUTOMATED SEGMENTATIONS:
       $p16_dir/batchX/segmentations/TaskY
       (here, TaskY is the task of the most recent model)

4. Upload the automated segmentations to the LUMC fileserver:
    \\vf-mdlz-onderzoekstraject\mdlz-onderzoekstraject$\MRI - segmentaties\Batch[X]\segmentaties_auto

5. Wait for medical experts to correct the scans.

6. After manual correction, upload the corrected segmentations from:
  \\vf-mdlz-onderzoekstraject\mdlz-onderzoekstraject$\MRI - segmentaties\BatchX\segmentaties_corr
  to:
  $nnUNet_raw_data_base/nnuNet_raw_data_base/Task[Y+1]/imagesTr
  (here, taskY+1 is the newly created task in step 3) 

7. Run ./run_training_pretrained.sh
   --> task = task[Y+1] 
   --> task_pretrained = task[Y]        or another model if you wish. but recommended is to use most recent model
   --> perform preprocessing = y
   Defaults should be fine!

