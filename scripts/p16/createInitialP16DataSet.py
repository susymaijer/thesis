#!/usr/bin/python
import os
import sys
import glob
import shutil
import random
from datetime import date
from collections import OrderedDict
import json
import SimpleITK as sitk
from pydicom import dcmread 
from pydicom.fileset import FileSet
import shared # our own

def get_id(dir):
    # List all the files
    files = os.listdir(dir)

    # Already files present
    if len(files) > 0:
        new = max([int(f.split("_")[1]) for f in files]) + 1 # Get the newest id
        new_str = "00000" + str(new)
        return new_str[-5:] # the id has 5 decimals

    # First one, so id is 00000
    else:
        return "00000"

def generate_dataset_json(overwrite_json_file, reference, task_dir, task, modality, labels,
                            tr_label_dir, ts_label_dir):
    json_file_exist = False
    json_path = os.path.join(task_dir, 'dataset.json')

    if os.path.exists(json_path):
        print(f'dataset.json already exist! {json_path}')
        json_file_exist = True

    if json_file_exist==False or overwrite_json_file:

        json_dict = OrderedDict()
        json_dict['name'] = f"Task{task}"
        json_dict['description'] = task
        json_dict['tensorImageSize'] = "3D"
        json_dict['reference'] = reference
        json_dict['licence'] = reference
        json_dict['release'] = "0.0"
        json_dict['modality'] = modality
        json_dict['labels'] = labels

        train_ids = os.listdir(tr_label_dir)
        test_ids = os.listdir(ts_label_dir)
        json_dict['numTraining'] = len(train_ids)

        #no modality in train image and labels in dataset.json
        json_dict['training'] = [{'image': f"./imagesTr/{i}", "label": f"./labelsTr/{i}"} for i in train_ids]
        json_dict['test'] = [f"./imagesTs/{i}" for i in test_ids]
        with open(json_path, 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=True)

if __name__ == "__main__":

    task=shared.initial_task

    # Define the paths to the folder containing the scans which we need to convert + the output dir
    task_name=f'Task{task}'
    img_dir=os.path.join(os.environ.get('nnUNet_raw_data_base'), 'p16', 'initial', 'scans')
    seg_dir=os.path.join(os.environ.get('nnUNet_raw_data_base'), 'p16', 'initial', 'labels')
    out_dir=os.path.join(os.environ.get('nnUNet_raw_data_base'), 'nnUNet_raw_data', task_name) 
    img_tr_dir=os.path.join(out_dir, 'imagesTr')
    img_ts_dir=os.path.join(out_dir, 'imagesTs')
    lab_tr_dir=os.path.join(out_dir, 'labelsTr')
    lab_ts_dir=os.path.join(out_dir, 'labelsTs')
    identify_path = os.path.join(out_dir, "identify.txt")

    # Create the output dir if it doesn't exist yet
    if (not(os.path.exists(img_dir)) or not(os.path.exists(seg_dir))):
        sys.exit(f"Directory {data_dir} does not exist. Abort program.")
    else:
        print(f"Cleaning {out_dir} and creating again")
        if(os.path.exists(out_dir)):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(img_tr_dir, exist_ok=True)
        os.makedirs(img_ts_dir, exist_ok=True)
        os.makedirs(lab_tr_dir, exist_ok=True)
        os.makedirs(lab_ts_dir, exist_ok=True)

    # Process each patient folder (either caseX or controlX)
    for case in os.listdir(img_dir):
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
            lab = sitk.ReadImage(os.path.join(seg_dir, case, p.split(os.sep)[-1], "segmentation.nii"))

            # Orient both to LPS
            img = sitk.DICOMOrient(img, 'RAS')
            lab = sitk.DICOMOrient(lab, 'RAS')
            #lab = sitk.Resample(lab, img)

            # Get the id for this scan
            scan_id = get_id(img_tr_dir)
            identify_record = f"Patient {patient}, folder {case}, date {d}, series {s}, created scan_id {scan_id}\n"
            f = open(identify_path, "a")
            f.write(identify_record)
            f.close()

            # Write the image
            print(identify_record)
            sitk.WriteImage(img, os.path.join(img_tr_dir, f'panc_{scan_id}_0000.nii.gz'))
            sitk.WriteImage(lab, os.path.join(lab_tr_dir, f'panc_{scan_id}.nii.gz'))

    print("Done")
    f = open(identify_path, "a")
    f.write("Done\n")
    f.close()

    # Create dataset.json
    generate_dataset_json(True, f"P16 {date.today()}", out_dir, task, {"0": "MRI"}, {"0": "background", "1": "pancreas"}, lab_tr_dir, lab_ts_dir)
