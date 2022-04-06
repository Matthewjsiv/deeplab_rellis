import yaml
import cv2
import matplotlib.pyplot as plt
import numpy as np

with open('Rellis_3D_ontology/ontology.yaml') as f:
    data = yaml.load(f)


print(data[1])


mask = cv2.imread('/home/matthew/Documents/image_seg_ws/dataset/train/targets/frame000000-1581624075_250.png').transpose(2, 0, 1)
print(mask.shape)

mask = mask.transpose(1,2,0)
print(mask.shape)
# plt.imshow(mask.transpose(1,2,0))
# plt.show()


convert = data[1]
data = data[2]
for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):
        key = mask[i,j,0]
        pallette = data[convert[key]]
        #brg
        mask[i,j,0] = pallette[0]
        mask[i,j,1] = pallette[1]
        mask[i,j,2] = pallette[2]
plt.imshow(mask)
plt.show()

# print(mask)
# locs = np.column_stack(np.where(mask == (3,3,3)))
# locs = locs[:,:-1]
# print(locs)
# print(locs.shape)
# print(locs[0])
# print(mask[locs[0][0],locs[0][1]])
# for elem in convert:
#     locs = np.column_stack(np.where(mask == (elem,elem,elem)))
#     if len(locs) > 0:
#         print(locs[0][0])
#         mask[locs[:][0],locs[:][1]] = (data[convert[elem]],data[convert[elem]],data[convert[elem]])
#     #mask[np.all(mask == (elem, elem, elem), axis=-1)] = (data[convert[elem]],data[convert[elem]],data[convert[elem]])
# plt.imshow(mask)
# plt.show()
