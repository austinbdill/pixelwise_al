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
        
        train_data = torchvision.datasets.Cityscapes('data', split="train", target_type="semantic", mode="fine", transform=input_trans, target_transform=target_trans)
        test_data = torchvision.datasets.Cityscapes('data', split="test", target_type="semantic", mode="fine", transform=input_trans, target_transform=target_trans)
        
        self.train_loader = torch.utils.data.DataLoader(train_data,batch_size=self.params["batch_size"])
        self.test_loader = torch.utils.data.DataLoader(test_data,batch_size=self.params["batch_size"])
        
        self.network = Simple().to(device="cuda")
        self.FC = FC().to(device="cuda")
        
        self.maskingnet = MaskingNet().to(device="cuda")
        
        self.parameters = list(self.maskingnet.parameters()) 
        self.parameters = list(self.network.parameters()) + list(self.FC.parameters()) + list(self.maskingnet.parameters())
        self.optimizer = optim.Adam(self.parameters)
        
        self.loss_fn = nn.MSELoss()
        
        self.regularizer = nn.L1Loss()
        
    def train(self):

        use_cuda = torch.cuda.is_available()
        print(use_cuda)
        # Generate directory for output
        date = datetime.datetime.now()
        self.result_dir = "results/" + date.strftime("%a_%b_%d_%I:%M%p")
        os.mkdir(self.result_dir)

        # Dump configurations for current run
        config_file = open(self.result_dir + "/configs.yml", "w+")
        yaml.dump(self.params, config_file)
        config_file.close()
        
        for e in range(self.params["n_epochs"]):
            
            #Training Loop
            print("TRAINING")
            for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                self.optimizer.zero_grad()
                
                images, segs = batch
                images = images.cuda()
                segs = segs.cuda()
                mask = self.maskingnet(images, e, i)
                sparse_segs = mask * segs
                feats = self.network(images, sparse_segs)
                pred_segs = self.FC(feats)
                loss = self.loss_fn(pred_segs, segs) + self.regularizer(mask, torch.zeros_like(mask))
                loss.backward()
                self.optimizer.step()
                
                if i == 1:
                    #torch.save(self.network.state_dict(), "cnn.pt")
                    #torch.save(self.maskingnet.state_dict(), "maskingnet.pt")
                    vutils.save_image(mask, self.result_dir + "/0_training_masks_" + str(e) + ".png", normalize=True)
                    #vutils.save_image(segs, self.result_dir + "/training_segs_" + str(e) + ".png", normalize=True)
                    vutils.save_image(pred_segs, self.result_dir + "/0_predicted_segs_" + str(e) + ".png", normalize=True)
                if i == 2:
                    vutils.save_image(mask, self.result_dir + "/2_training_masks_" + str(e) + ".png", normalize=True)
                    #vutils.save_image(segs, self.result_dir + "/2_training_segs_" + str(e) + ".png", normalize=True)
                    vutils.save_image(pred_segs, self.result_dir + "/2_predicted_segs_" + str(e) + ".png", normalize=True)
                if i == 5:
                    vutils.save_image(mask, self.result_dir + "/5_training_masks_" + str(e) + ".png", normalize=True)
                    #vutils.save_image(segs, self.result_dir + "/5_training_segs_" + str(e) + ".png", normalize=True)
                    vutils.save_image(pred_segs, self.result_dir + "/5_predicted_segs_" + str(e) + ".png", normalize=True)
                if i == 10:
                    vutils.save_image(mask, self.result_dir + "/10_training_masks_" + str(e) + ".png", normalize=True)
                    #vutils.save_image(segs, self.result_dir + "/10_training_segs_" + str(e) + ".png", normalize=True)
                    vutils.save_image(pred_segs, self.result_dir + "/10_predicted_segs_" + str(e) + ".png", normalize=True)
            print("TESTING")
            epoch_loss = 0.0
            for i, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                images, segs = batch
                images = images.cuda()
                segs = segs.cuda()
                mask = self.maskingnet(images, 0, 0)
                sparse_segs = mask * segs
                feats = self.network(images, sparse_segs)
                pred_segs = self.FC(feats)
                loss = self.loss_fn(pred_segs, segs)
                epoch_loss = epoch_loss + loss.detach().cpu().numpy()
            print(epoch_loss / len(self.test_loader))
