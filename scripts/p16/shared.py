import json
import os
from collections import OrderedDict

initial_task = 610 # don't change! this was used for the creation of the first batch. 
p16_dir = os.path.join(os.environ.get('nnUNet_raw_data_base'), 'p16')

def getUserspecificEnvironmentVariables(filename):
    import importlib
    f = open(filename)
    global data
    data = imp.load_source('data', '', f)
    f.close()
    return data

def determine_task_id():
    base=os.path.join(os.environ.get('nnUNet_raw_data_base'), 'nnUNet_raw_data')
    return max([int(x[4:]) for x in os.listdir(base) if x.startswith('Task6')])

def determine_batch_id():
    base=os.path.join(os.environ.get('nnUNet_raw_data_base'), 'p16')
    return max([int(x[5:]) for x in os.listdir(base) if x.startswith('batch')])

def get_scan_id(dir_name):
    # List all the files
    files = os.listdir(dir_name)

    # Already files present
    if len(files) > 0:
        new = max([int(f.split("_")[1]) for f in files]) + 1 # Get the newest id
        new_str = "00000" + str(new)
        return new_str[-5:] # the id has 5 decimals

    # First one, so id is 00000
    else:
        return "00000"

def generate_dataset_json(overwrite_json_file, reference, task_dir, task, modality, labels,
                            tr_dir, ts_dir, is_label_dir=True):
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

        train_ids = os.listdir(tr_dir)
        test_ids = os.listdir(ts_dir)
        json_dict['numTraining'] = len(train_ids)

        #no modality in train image and labels in dataset.json
        if is_label_dir:
           json_dict['training'] = [{'image': f"./imagesTr/{i}", "label": f"./labelsTr/{i}"} for i in train_ids]
           json_dict['test'] = [f"./imagesTs/{i}" for i in test_ids]
        else:
            train_ids = [i[:10] + i[15:] for i in train_ids]
            test_ids = [i[:10] + i[15:] for i in test_ids]
            json_dict['training'] = [{'image': f"./imagesTr/{i}", "label": f"./labelsTr/{i}"} for i in train_ids]
            json_dict['test'] = [f"./imagesTs/{i}" for i in test_ids]

        with open(json_path, 'w') as f:
            json.dump(json_dict, f, indent=4, sort_keys=True)



