#DATASET: https://unmannedlab.github.io/research/RELLIS-3D
#download the full images, annotations, and split files
#create an empty dataset folder (might also need to create train, test, and validation subfolders)


import shutil
import os
cwd = os.getcwd()

f = open("train.lst","r")
lines = f.readlines()
for line in lines:
    input,target = line.split(' ')
    target = target.replace('\n','')
    shutil.copy('Rellis_3D_pylon_camera_node/Rellis-3D/' + input, 'dataset/train/input')
    shutil.copy('Rellis_3D_pylon_camera_node_label_id/Rellis-3D/' + target, 'dataset/train/targets')
f.close()

f = open("test.lst","r")
lines = f.readlines()
for line in lines:
    input,target = line.split(' ')
    target = target.replace('\n','')
    shutil.copy('Rellis_3D_pylon_camera_node/Rellis-3D/' + input, 'dataset/test/input')
    shutil.copy('Rellis_3D_pylon_camera_node_label_id/Rellis-3D/' + target, 'dataset/test/targets')
f.close()

f = open("val.lst","r")
lines = f.readlines()
for line in lines:
    input,target = line.split(' ')
    target = target.replace('\n','')
    shutil.copy('Rellis_3D_pylon_camera_node/Rellis-3D/' + input, 'dataset/validation/input')
    shutil.copy('Rellis_3D_pylon_camera_node_label_id/Rellis-3D/' + target, 'dataset/validation/targets')
f.close()
