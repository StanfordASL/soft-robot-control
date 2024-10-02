# This file automates copying the models from /media/lpabon/Backup Plus/jonas_soft_robot_data/trunk_adiabatic_10ms_N=100
# to /home/lpabon/Documents/6D Adiabatic Models
# The source directory contains 100 folders called '000' to '099' 
# Inside each of these, we want to copy everything except the folders: "decay" and "open-loop"


import os
import shutil
from os import listdir
from os.path import isdir, join

# Source directory
source_dir = "/media/lpabon/Backup Plus/jonas_soft_robot_data/trunk_adiabatic_10ms_N=100"

# Destination directory
destination_dir = "/home/lpabon/Documents/6D Adiabatic Models"

# Get all the folders in the source directory
folders = [name for name in sorted(listdir(source_dir)) if isdir(join(source_dir, name))]

# Folders in source directory:  ['000', '001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '052', '053', '054', '055', '056', '057', '058', '059', '060', '061', '062', '063', '064', '065', '066', '067', '068', '069', '070', '071', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '084', '085', '086', '087', '088', '089', '090', '091', '092', '093', '094', '095', '096', '097', '098', '099']

# Copy all the folders from source to destination, except the folders "decay" and "open-loop" inside each folder

for folder in folders:
    source_folder = join(source_dir, folder)
    destination_folder = join(destination_dir, folder)
    if not isdir(destination_folder):
        os.makedirs(destination_folder)
    for item in listdir(source_folder):
        if item not in ["decay", "open-loop"]:
            source_item = join(source_folder, item)
            destination_item = join(destination_folder, item)
            if isdir(source_item):
                shutil.copytree(source_item, destination_item)
            else:
                shutil.copy2(source_item, destination_item)

print("Done copying files")