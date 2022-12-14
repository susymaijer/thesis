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

4. Upload the following to the LUMC fileserver:
   a. NIFTIS
   shark location:  $p16_dir/batchX/niftis
   lumc fileserver:  \\vf-mdlz-onderzoekstraject\mdlz-onderzoekstraject$\MRI - segmentaties\Batch[X]\scans

   b. AUTOMATED SEGMENTATIONS
   shark location:  $p16_dir/batchX/segmentations/TaskY
   lumc fileserver:   \\vf-mdlz-onderzoekstraject\mdlz-onderzoekstraject$\MRI - segmentaties\Batch[X]\segmentaties_auto
  
   c. KEY FILE
   shark location:  $p16_dir/batchX/identify.txt
   lumc fileserver: \\vf-mdlz-onderzoekstraject\mdlz-onderzoekstraject$\MRI - segmentaties\Batch[X]

5. Wait for medical experts to correct the scans.

6. After manual correction, upload the corrected segmentations from:
  \\vf-mdlz-onderzoekstraject\mdlz-onderzoekstraject$\MRI - segmentaties\BatchX\segmentaties_corr
  to:
  $p16_dir/batchX/corr

7. Run ./1_evaluateCorrectionsAndStartNewTraining.sh
   --> evaluates the predictions by comparing them to the corrections
   --> copies the corrections to the task created by step 3
   --> starts a training run for this new task (including preprocessing step)
