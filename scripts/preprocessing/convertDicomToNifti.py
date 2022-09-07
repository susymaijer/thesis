#!/usr/bin/python
import os
import sys
import dicom2nifti
import SimpleITK as sitk
import shutil

if __name__ == "__main__":

    # Define the paths to the folder containing the scans which we need to convert + the output dir
    folder=sys.argv[1]
    base_dir=os.path.join(os.environ.get('nnUNet_raw_data_base'), 'alex', folder)
    data_dir=os.path.join(base_dir, 'original')

    # Create the output dir if it doesn't exist yet
    if (not(os.path.exists(data_dir))):
        sys.exit(f"Directory {data_dir} does not exist. Abort program.")
    else:
        out_dir=os.path.join(base_dir, 'scans')
        os.makedirs(out_dir, exist_ok=True)
        print(f"Created {out_dir}")

    # Convert each scan 
    for scan_dir in os.listdir(data_dir):
        print(f"Converting {scan_dir} to nifti")
        curr_dir=os.path.join(data_dir,scan_dir)
        dicom2nifti.convert_directory(curr_dir, out_dir, compression=True)

        # Find nifti file and rename
        nifti_name=[name for name in os.listdir(out_dir) if not(name.startswith('panc_'))][0]
        new_name=f"panc_{scan_dir}_0000.nii.gz"
        print(f"Renaming from {nifti_name} to {new_name}")
        new_path=os.path.join(out_dir,new_name)
        os.rename(os.path.join(out_dir,nifti_name), new_path)

