#!/usr/bin/python
import os
import sys
import glob
import shutil
import random
import zipfile
from datetime import date
from collections import OrderedDict
import SimpleITK as sitk
from pydicom import dcmread 
from pydicom.fileset import FileSet
import shared # our own

def determine_new_task_id():
    base=os.path.join(os.environ.get('nnUNet_raw_data_base'), 'nnUNet_raw_data')
    return max([int(x[4:]) for x in os.listdir(base) if x.startswith('Task6')]) + 1

if __name__ == "__main__":

    # Get the batch id and task id
    batch=shared.determine_batch_id()
    task=determine_new_task_id()    
    print(f"We're going to create task {task} from batch {batch}. Is this OK? Enter [y/n].")
    answer=input()
    if answer != "y":
        sys.exit("Abort")

    # Define the paths to the folder containing the scans which we need to convert + the output dir
    task_name=f'Task{task}'
    img_dir=os.path.join(shared.p16_dir, f'batch{batch}')
    out_niftis_dir=os.path.join(img_dir, 'niftis')
    out_dir=os.path.join(os.environ.get('nnUNet_raw_data_base'), 'nnUNet_raw_data', task_name)
    img_tr_dir=os.path.join(out_dir, 'imagesTr')
    img_ts_dir=os.path.join(out_dir, 'imagesTs')
    lab_tr_dir=os.path.join(out_dir, 'labelsTr')
    lab_ts_dir=os.path.join(out_dir, 'labelsTs')
    identify_path = os.path.join(out_dir, "identify.txt")
    os.mkdir(out_niftis_dir)

    # Copy contents of previous task to new task
    if (not(os.path.exists(img_dir))):
        sys.exit(f"Directory {img_dir} does not exist. Abort program.")
    else:
        # Add the data of the old task to the new directory
        prev_task_dir=os.path.join(os.environ.get('nnUNet_raw_data_base'), 'nnUNet_raw_data', f'Task{task-1}')
        shutil.copytree(prev_task_dir, out_dir)
        print(f"Copied contents of {prev_task_dir} to new task dir\n")
    
    # Unzip the folder from the LUMC cluster and get paths to files
    print("Unzipping LUMC folder..\n")
    img_zip=[x for x in os.listdir(img_dir) if x.endswith(".zip")][0]
    with zipfile.ZipFile(os.path.join(img_dir, img_zip),"r") as zip_ref:
       zip_ref.extractall(img_dir)
    img_dir=os.path.join(img_dir, img_zip.split(".zip")[0])

    # Process each patient folder (either caseX or controlX)
    for case in os.listdir(img_dir):
        print(f"Processing {case}")
        # Read the DICOMDIR file
        ds = dcmread(os.path.join(img_dir, case, "DICOMDIR"))
        fs = FileSet(ds)
        seq = ds[0x0004, 0x1220] # Directory Record Sequence

        # Get the paths to the T2 series (we also have Dixon scans)
        new_T2_series = False
        paths = []
        for i, x in enumerate(seq):
            type = x[0x0004, 0x1430].value
            
            # Patient data
            if type == "PATIENT":
                patient = x[0x0010, 0x0020].value # Anonymised patient id

            # Studies date
            if type == "STUDY":
                study_date = x[0x0008, 0x0020].value

            # Check whether a new T2 series started
            if type == "SERIES":
                series_desc = x[0x0008, 0x103e].value # Series description
                # T2 series, so we want this series
                if series_desc.startswith("T2"):
                    new_T2_series = series_desc
                    continue

            # Get the root directory of the new T2 series
            if new_T2_series:
                p = os.path.join(fs.path, *x[0x0004, 0x1500].value[:-1])
                paths.append((p, study_date, new_T2_series))
                new_T2_series = False

        # Now read the DICOM and segmentations and convert
        for p, d, s in paths:
            # Get first file in the scan directory
            file_name = glob.glob(p + os.sep + "*")[0]
            file_reader = sitk.ImageFileReader()
            file_reader.SetFileName(file_name)
            file_reader.ReadImageInformation()

            # Loads the files in the correct order
            series_ID = file_reader.GetMetaData('0020|000e')
            sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(p, series_ID)

            # Read the image and segmentation
            img = sitk.ReadImage(sorted_file_names)

            # Orient both to LPS
            img = sitk.DICOMOrient(img, 'RAS')

            # Get the id for this scan
            scan_id = shared.get_scan_id(img_tr_dir)
            identify_record = f"Patient {patient}, folder {case}, date {d}, series {s}, created scan_id {scan_id}\n"
            f = open(identify_path, "a")
            f.write(identify_record)
            f.close()

            # Write the image
            print(identify_record)
            sitk.WriteImage(img, os.path.join(img_tr_dir, f'panc_{scan_id}_0000.nii.gz'))
            sitk.WriteImage(img, os.path.join(out_niftis_dir, f'panc_{scan_id}_0000.nii.gz'))

    print(f"Done")
    f = open(identify_path, "a")
    f.write(f"Done batch {batch}\n")
    f.close()

    # Create dataset.json
    shared.generate_dataset_json(True, f"P16 {date.today()} batch{batch}", out_dir, task, {"0": "MRI"}, {"0": "background", "1": "pancreas"}, img_tr_dir, img_ts_dir, False)
