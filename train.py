#DATASET: https://unmannedlab.github.io/research/RELLIS-3D
#see split_dataset.py for directory setup

#https://expoundai.wordpress.com/2019/08/30/transfer-learning-for-segmentation-using-deeplabv3-in-pytorch/

# cd ~/.cache/torch/checkpoints/
# wget -c --no-check-certificate https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth
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

# create a color pallette, selecting a color for each class
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

        # image = cv2.imread(
        #     img_name, self.imagecolorflag).transpose(2, 0, 1)
        input_image = Image.open(img_name)
        preprocess = transforms.Compose([
            transforms.CenterCrop((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        image = input_tensor
        #print(image.shape)
        #input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


        msk_name = self.mask_names[idx]
        # mask = torch.Tensor(cv2.imread(msk_name).transpose(2, 0, 1))
        # mask = mask.to(torch.uint8)
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

        #print(mask.shape)

        sample = {'image': image, 'mask': mask}
        return sample


def DeepLabModel(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet50(
        pretrained=True, progress=True)
    #model = torch.hub.load('pytorch/vision:v0.5.0', 'deeplabv3_resnet101', pretrained=True)
    # Added a Sigmoid activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    return model

def train_model(model, dataloader,num_items, val_loader, num_val_items,batchsize=2,val_batchsize=2,num_epochs=10):
    since = time.time()

    # Specify the loss function
    criterion = torch.nn.CrossEntropyLoss()
    # Specify the optimizer
    #change to classifier.parameters
    #adjust learning rate, increase weight decay
    #TEMP NOTES - need to try either raising learning rate again or freezing more params
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)

    FREEZE = 8
    ct = 0
    for child in model.children():
        for subchild in child.children():
            ct += 1
            if ct < FREEZE:
                print(subchild)
                for param in subchild.parameters():
                    param.requires_grad = False
    #oldish + model.classifier.parameters()
    # for child in model.children():
    #     for parameter in child.parameters():
    #         parameter.requires_grad = False
    #     break #only wanna do this once


    writer = SummaryWriter('freeze_' + str(FREEZE))
    idx = 0
    idxv = 0
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        train_loss = 0
        val_loss = 0

        ###################TRAINING
        model.train()
        for sample in tqdm(iter(dataloader),total=num_items/batchsize):
            inputs = sample['image'].cuda()
            masks = sample['mask'].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            print(inputs.shape)
            outputs = model(inputs)
            outputs = outputs['out']
            outputs = torch.flatten(outputs,start_dim=2, end_dim=3)

            masks = torch.flatten(masks, start_dim=1)
            masks = masks.to(torch.int64)

            # print(masks.shape)
            # print(outputs.shape)
            loss = criterion(outputs, masks)
            writer.add_scalar('Loss/train', loss.item(),idx)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # if idx % 1000 == 0:
            # torch.save(model.state_dict(), 'checkpoints/freeze-8/checkpoint-' + str(idx))
            idx+=1

        #save every epoch
        torch.save(model.state_dict(), 'checkpoints/freeze-' + str(FREEZE) + '/checkpoint-' + str(epoch))

        ###################VALIDATION
        model.eval()
        with torch.no_grad():
            for sample in tqdm(iter(val_loader),total=num_val_items/val_batchsize):
                inputs = sample['image'].cuda()
                masks = sample['mask'].cuda()

                #print(inputs.shape)
                outputs = model(inputs)
                outputs = outputs['out']
                outputs = torch.flatten(outputs,start_dim=2, end_dim=3)

                masks = torch.flatten(masks, start_dim=1)
                masks = masks.to(torch.int64)

                # print(masks.shape)
                # print(outputs.shape)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                idxv+=1

        #compute avg
        train_loss = train_loss/num_items
        val_loss = val_loss/num_val_items

        writer.add_scalar('Epoch_Loss/train', train_loss,epoch)
        writer.add_scalar('Epoch_Loss/validation', val_loss,epoch)

        print('epoch_loss_train = ' + str(train_loss))
        print('val_loss_train = ' + str(val_loss))


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return



dataset = SegDataset('dataset/train','input','targets')
v_dataset = SegDataset('dataset/validation','input','targets')
#print(dataset[0])
# plt.imshow(transforms.ToPILImage()(dataset[0]['image']))
# plt.show()

dataloader = DataLoader(dataset, batch_size=2)
v_dataloader = DataLoader(v_dataset, batch_size=4)

model = DeepLabModel(20)
# for elem in model.backbone.conv1.parameters():
#     print(elem)

#model.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

#model.load_state_dict(torch.load("checkpoints/freeze-8/checkpoint-20000"))
print(model)
model.train()
model.cuda()

train_model(model,dataloader,len(dataset),v_dataloader,len(v_dataset),batchsize=2,val_batchsize=4,num_epochs=10)




#
# tensor = dataset[0]['image'].unsqueeze(0).cuda()
# print(tensor.shape)
#
#
# model.eval()
# output = model(tensor)
# output = output['out'][0]
#print(output)
#
#
#
#
#
#
#
#
# test = dataset[0]['mask']
# nmask = np.zeros((1024,1024,3),dtype=np.int32)
# for i in range(test.shape[0]):
#     for j in range(test.shape[1]):
#         key = test[i,j].byte().cpu().item()
#         pallette = DATA[2][key]
#
#         nmask[i,j,0] = pallette[0]
#         nmask[i,j,1] = pallette[1]
#         nmask[i,j,2] = pallette[2]
# #
# plt.imshow(nmask)
# plt.show()
#
# test = dataset[0]['mask'].flatten()
# # output = torch.flatten(output,start_dim=1, end_dim=2).transpose(0,1)
# # #output = output.to(torch.int64)
# # test = test.to(torch.int64).cuda()
# # print(test.shape)
# # print(output.shape)
# # criterion = torch.nn.CrossEntropyLoss()(output,test)
# # print(criterion.item())
# # print(test)
#
#
#
#
#
# output = output.argmax(0)
# print(output)
# # plot the semantic segmentation predictions of 21 classes in each color
# mask = output.byte().cpu()
# mask.unsqueeze_(-1)
# mask = mask.expand(1024,1024,3)
# mask = mask.numpy()
# mask = mask.astype(np.int32)
# print(mask.shape)
# convert = DATA[1]
# data = DATA[2]
# print(data)
# print(mask.shape)
# #print(mask)
# for i in range(mask.shape[0]):
#     for j in range(mask.shape[1]):
#         key = mask[i,j,0]
#         pallette = data[key]
#
#         mask[i,j,0] = pallette[0]
#         mask[i,j,1] = pallette[1]
#         mask[i,j,2] = pallette[2]
#
#
# #r = Image.fromarray(mask).resize((1024,1024))
# plt.imshow(mask)
# plt.show()
