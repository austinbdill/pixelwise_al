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
        #self.maskingnet = MaskingNet()
        
        parameters = list(self.network.parameters()) + list(self.FC.parameters())
        self.optimizer = optim.Adam(parameters)
        
        self.loss_fn = nn.L1Loss()
        
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
                #mask = self.maskingnet(images)
                #sparse_segs = mask * segs
                sparse_segs = segs
                feats = self.network(images, sparse_segs)
                pred_segs = self.FC(feats)
                loss = self.loss_fn(pred_segs, segs)
                loss.backward()
                self.optimizer.step()
                epoch_loss = epoch_loss + loss.detach().numpy()
                
                if i == 1:
                    torch.save(self.network.state_dict(), "cnn_25.pt")
                    vutils.save_image(images, self.result_dir + "/training_images_" + str(e) + ".png")
                    vutils.save_image(segs, self.result_dir + "/training_segs_" + str(e) + ".png", normalize=True)
                    vutils.save_image(pred_segs, self.result_dir + "/predicted_segs_" + str(e) + ".png", normalize=True)
            print(epoch_loss / len(self.train_loader))