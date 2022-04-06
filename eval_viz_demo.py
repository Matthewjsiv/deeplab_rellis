from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import cv2
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import time
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm

#Use my ontology file, not theirs
with open('Rellis_3D_ontology/ontology.yaml') as f:
    DATA = yaml.load(f)

class SegDataset(Dataset):
    """Segmentation Dataset"""

    def __init__(self, root_dir, imageFolder, maskFolder):
        """
        Args:
            root_dir (string): Directory with all the images and should have the following structure.
            root
            --Images
            -----Img 1
            -----Img N
            --Mask
            -----Mask 1
            -----Mask N
            imageFolder (string) = 'Images' : Name of the folder which contains the Images.
            maskFolder (string)  = 'Masks : Name of the folder which contains the Masks.
        """
        self.root_dir = root_dir
        self.image_names = sorted(
            glob.glob(os.path.join(self.root_dir, imageFolder, '*')))
        self.mask_names = sorted(
            glob.glob(os.path.join(self.root_dir, maskFolder, '*')))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        input_image = Image.open(img_name)
        preprocess = transforms.Compose([
            transforms.CenterCrop((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        image = input_tensor

        #we also want the normal image for visualization purposes
        preprocess = transforms.Compose([
            transforms.CenterCrop((1024, 1024)),
            transforms.ToTensor()])
        nimage = preprocess(input_image)


        msk_name = self.mask_names[idx]
        mask = Image.open(msk_name)
        preprocess = transforms.Compose([
            transforms.CenterCrop((1024, 1024)),
            transforms.ToTensor()
        ])
        mask = preprocess(mask) * 255
        mask = mask.to(torch.uint8)
        mask = mask.squeeze()

        #convert keys 0:34 to 0:19
        convert = DATA[1]
        for elem in convert:
            mask[mask==elem] = convert[elem]


        sample = {'image': image, 'mask': mask, 'nimage': nimage}
        return sample

def DeepLabModel(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet50(
        pretrained=True, progress=True)

    model.classifier = DeepLabHead(2048, outputchannels)
    return model

#you can vizualize different directories by changing the directory names here
dataset = SegDataset('dataset/test','input','targets')
dataloader = DataLoader(dataset, batch_size=4)
num_items = len(dataset)

model = DeepLabModel(20)
#replace this with your checkpoint
model.load_state_dict(torch.load("checkpoints/freeze-8_oldish-5/checkpoint-20000"))
model.eval()
model.cuda()

for child in model.children():
    print(child)
    break



writer = SummaryWriter('vizboard')

with torch.no_grad():
    idx = 0
    for sample in tqdm(dataset,total=num_items):
        fig=plt.figure(dpi=400)
        nimage = sample['nimage'].transpose(0,2).transpose(0,1)
        #writer.add_image('sample'+ str(idx) + '/image',nimage,idx,dataformats='CHW')
        fig.add_subplot(1, 3, 1).title.set_text('Image')
        plt.imshow(nimage)
        plt.axis('off')

        mask = sample['mask']
        nmask = np.zeros((1024,1024,3),dtype=np.int32)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                key = mask[i,j].byte().cpu().item()
                pallette = DATA[2][key]

                nmask[i,j,0] = pallette[0]
                nmask[i,j,1] = pallette[1]
                nmask[i,j,2] = pallette[2]
        #writer.add_image('sample'+ str(idx) + '/GT',nmask,idx,dataformats='HWC')
        fig.add_subplot(1, 3, 2).title.set_text('GT')
        plt.imshow(nmask)
        plt.axis('off')

        input = sample['image'].unsqueeze(0).cuda()
        output = model(input)
        output = output['out'][0]

        output = output.argmax(0)
        # plot the semantic segmentation predictions of 21 classes in each color
        output = output.byte().cpu()
        output.unsqueeze_(-1)
        output = output.expand(1024,1024,3)
        output = output.numpy()
        output = output.astype(np.int32)
        convert = DATA[1]
        data = DATA[2]
        #print(mask)
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                key = output[i,j,0]
                pallette = data[key]

                output[i,j,0] = pallette[0]
                output[i,j,1] = pallette[1]
                output[i,j,2] = pallette[2]
        #writer.add_image('sample'+ str(idx) + '/Prediction',output,idx,dataformats='HWC')
        fig.add_subplot(1, 3, 3).title.set_text('Prediction')
        plt.imshow(output)
        plt.axis('off')

        idx +=1
        plt.imshow(output)
        plt.savefig('viz/test/' + str(idx) + '.png')
writer.close()
