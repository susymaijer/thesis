#!/usr/bin/python
import os
import sys
import shutil
import random
from collections import OrderedDict
import json

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

    # Define the paths to the folder containing the scans which we need to convert + the output dir
    task='Task540'
    data_dir=os.path.join(os.environ.get('nnUNet_raw_data_base'), 'totalsegmentator', 'TotalsegmentatorPancreas_dataset')
    out_dir=os.path.join(os.environ.get('nnUNet_raw_data_base'), 'nnUNet_raw_data', task) 
    img_tr_dir=os.path.join(out_dir, 'imagesTr')
    img_ts_dir=os.path.join(out_dir, 'imagesTs')
    lab_tr_dir=os.path.join(out_dir, 'labelsTr')
    lab_ts_dir=os.path.join(out_dir, 'labelsTs')

    # Create the output dir if it doesn't exist yet
    if (not(os.path.exists(data_dir))):
        sys.exit(f"Directory {data_dir} does not exist. Abort program.")
    else:
        print(f"Cleaning {out_dir} and creating again")
        shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(img_tr_dir, exist_ok=True)
        os.makedirs(img_ts_dir, exist_ok=True)
        os.makedirs(lab_tr_dir, exist_ok=True)
        os.makedirs(lab_ts_dir, exist_ok=True)

    # Determine which scans we use for test set
    scans=[x for x in os.listdir(data_dir) if x.startswith('s')]
    test_size= int(0.15 * len(scans))
    test_filenames = [i for i in random.sample(scans, k=test_size)]
    print(f"Create test set of {len(test_filenames)} images (total: {len(scans)})")

    # Copy each scan to desired location
    for i, scan in enumerate(scans):
        print(f"Processed {i} out of {len(scans)}")
        # Get paths to our data
        scan_img=os.path.join(data_dir, scan, 'ct.nii.gz')
        scan_lab=os.path.join(data_dir, scan, 'pancreas.nii.gz')

        # Get output paths (either ..Ts or ..Tr)
        if scan in test_filenames: 
            out_img=os.path.join(img_ts_dir, f'panc_{scan}_0000.nii.gz')
            out_lab=os.path.join(lab_ts_dir, f'panc_{scan}.nii.gz')
        else:
            out_img=os.path.join(img_tr_dir, f'panc_{scan}_0000.nii.gz')
            out_lab=os.path.join(lab_tr_dir, f'panc_{scan}.nii.gz')

        # Copy
        shutil.copy(scan_img, out_img)
        shutil.copy(scan_lab, out_lab)

    # Create dataset.json
    generate_dataset_json(True, "TotalSegmentator", out_dir, 540, {"0": "CT"}, {"0": "background", "1": "pancreas"}, lab_tr_dir, lab_ts_dir)
