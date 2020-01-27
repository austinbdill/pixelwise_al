import os
import yaml
import datetime
import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
from torch import optim
import torchvision.utils as vutils

from models.networks import *

class BaseTrainer(object):

    def __init__(self, params):

        self.params = params
        
        input_trans = transforms.Compose([transforms.CenterCrop(300), transforms.Resize(100), transforms.ToTensor()])
        target_trans = transforms.Compose([transforms.CenterCrop(300),  transforms.Resize(100), transforms.ToTensor()])
        
        data = torchvision.datasets.Cityscapes('data', split="train", target_type="semantic", mode="fine", transform=input_trans, target_transform=target_trans)
        
        self.train_loader = torch.utils.data.DataLoader(data,batch_size=self.params["batch_size"])
        
        self.network = Simple()
        self.FC = FC()
        
        self.maskingnet = MaskingNet()
        
        self.parameters = list(self.maskingnet.parameters()) #list(self.network.parameters()) + list(self.FC.parameters()) + list(self.maskingnet.parameters())
        self.optimizer = optim.Adam(self.parameters)
        
        self.loss_fn = nn.L1Loss()
        
        self.regularizer = nn.L1Loss()
        
    def train(self):

        # Generate directory for output
        date = datetime.datetime.now()
        self.result_dir = "results/" + date.strftime("%a_%b_%d_%I:%M%p")
        os.mkdir(self.result_dir)

        # Dump configurations for current run
        config_file = open(self.result_dir + "/configs.yml", "w+")
        yaml.dump(self.params, config_file)
        config_file.close()
        
        for e in range(self.params["n_epochs"]):
            
            epoch_loss = 0.0
            #Training Loop
            for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                self.optimizer.zero_grad()
                
                images, segs = batch
                #depth_image = torch.cat((images, segs), 1)
                mask = self.maskingnet(images, i)
                sparse_segs = mask * segs
                #print(mask[0])
                #sparse_segs = segs
                feats = self.network(images, sparse_segs)
                pred_segs = self.FC(feats)
                loss = self.regularizer(mask, torch.zeros_like(mask)) + self.loss_fn(pred_segs, segs)
                loss.backward()
                self.optimizer.step()
                epoch_loss = epoch_loss + loss.detach().numpy()
                
                if i == 1:
                    #torch.save(self.network.state_dict(), "cnn.pt")
                    #torch.save(self.maskingnet.state_dict(), "maskingnet.pt")
                    vutils.save_image(sparse_segs, self.result_dir + "/training_masks_" + str(e) + ".png", normalize=True)
                    vutils.save_image(segs, self.result_dir + "/training_segs_" + str(e) + ".png", normalize=True)
                    vutils.save_image(pred_segs, self.result_dir + "/predicted_segs_" + str(e) + ".png", normalize=True)
                if i == 2:
                    vutils.save_image(sparse_segs, self.result_dir + "/2_training_masks_" + str(e) + ".png", normalize=True)
                    vutils.save_image(segs, self.result_dir + "/2_training_segs_" + str(e) + ".png", normalize=True)
                    vutils.save_image(pred_segs, self.result_dir + "/2_predicted_segs_" + str(e) + ".png", normalize=True)
                if i == 5:
                    vutils.save_image(sparse_segs, self.result_dir + "/5_training_masks_" + str(e) + ".png", normalize=True)
                    vutils.save_image(segs, self.result_dir + "/5_training_segs_" + str(e) + ".png", normalize=True)
                    vutils.save_image(pred_segs, self.result_dir + "/5_predicted_segs_" + str(e) + ".png", normalize=True)
                if i == 10:
                    vutils.save_image(sparse_segs, self.result_dir + "/10_training_masks_" + str(e) + ".png", normalize=True)
                    vutils.save_image(segs, self.result_dir + "/10_training_segs_" + str(e) + ".png", normalize=True)
                    vutils.save_image(pred_segs, self.result_dir + "/10_predicted_segs_" + str(e) + ".png", normalize=True)
            print(epoch_loss / len(self.train_loader))