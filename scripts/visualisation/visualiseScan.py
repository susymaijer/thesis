#!/usr/bin/python
import os
import sys
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
# ipywidgets for some interactive plots
from ipywidgets.widgets import * 

#### administrative stuff

def get_filename_from_rel_id(img_path, rel_patient_id):
    return os.listdir(img_path)[rel_patient_id]

def get_patient_id_from_filename(filename):
    file_name = filename.split(".")[0]
    parts = file_name.split("_")
    return parts[1]

def get_patient_id_from_rel_id(img_path, rel_patient_id):
    filename = get_filename_from_rel_id(img_path, rel_patient_id)
    return get_patient_id_from_filename(filename)

def get_filename(patient_id, modality= None):
    if modality:
        filename = f'/panc_{patient_id}_{modality}.nii.gz'
    else:
        filename = f'/panc_{patient_id}.nii.gz'
    return filename

def get_file(img_path, patient_id, modality = None):
    filename = get_filename(patient_id, modality)
    return nib.load(img_path + filename)

def get_3d_array(img_path, patient_id, modality = None):
    return np.array(get_file(img_path, patient_id, modality).dataobj)

#### visualisation functions

def visualise(x):
    img_slice = img[:,:,x]
    seg_slice = seg[:,:,x]
    print(f'Dimensions: {img.shape}')

    # show label single slice
    f = plt.figure(figsize=(30,30))
    a1 = f.add_subplot(2, 2, 1)
    plt.imshow(seg_slice, cmap=plt.cm.gray)

    # show label over dicom
    a1 = f.add_subplot(2, 2, 2)
    plt.imshow(img_slice, cmap=plt.cm.gray)
    plt.imshow(seg_slice, cmap=plt.cm.jet_r, alpha=0.1)

    # show label over dicom
    a1 = f.add_subplot(2, 2, 3)
    plt.imshow(img_slice, cmap=plt.cm.gray)
    plt.show()

    return x

if __name__ == "__main__":

    # Define the paths to the folder containing the scans which we need to convert + the output dir
    folder = sys.argv[1]
    task = sys.argv[2]
    rel_id = int(sys.argv[3])
    base_dir = os.path.join(os.environ.get('nnUNet_raw_data_base'), 'alex', folder)
    scan_dir = os.path.join(base_dir, 'scans')
    seg_dir = os.path.join(base_dir, 'segmentations', task)
    modality = "0000"

    # Load the scan and the segmentation 
    id = get_patient_id_from_rel_id(scan_dir, rel_id)
    print(f"Loading scan {id} from folder {folder}")
    global img = get_3d_array(scan_dir, id, modality)
    global seg = get_3d_array(seg_dir, id)

