import os
import yaml
import datetime
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from data.data_utils import *
from tqdm import tqdm

class BaseTrainer(object):

    def __init__(self, params):

        self.params = params
        
        input_transform = transforms.Compose([RandomCrop(350), Rescale(100), ToTensor()])
        
        train_data = KittiDataset("data/kitti/train", transform=input_transform)
        test_data = KittiDataset("data/kitti/val", transform=input_transform)
        
        self.train_loader = torch.utils.data.DataLoader(train_data,batch_size=self.params["batch_size"])
        self.test_loader = torch.utils.data.DataLoader(test_data,batch_size=self.params["batch_size"])
        
    def train(self):
        # Generate directory for output
        date = datetime.datetime.now()
        self.result_dir = "results/" + self.params["type"] + "_" + date.strftime("%a_%b_%d_%I:%M%p")
        os.mkdir(self.result_dir)

        # Dump configurations for current run
        config_file = open(self.result_dir + "/configs.yml", "w+")
        yaml.dump(self.params, config_file)
        config_file.close()
        
        self.writer = SummaryWriter(self.result_dir)
        
        for e in range(self.params["n_epochs"]):
            
            #Training Loop
            print("TRAINING")
            #print(self.tau)
            for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                #training_loss, avg_mask, critic_loss = self.training_step(batch, e, i)
                training_loss, avg_mask = self.training_step(batch, e, i)
                
                if i % 100 == 0:
                    step = e*len(self.train_loader)+i
                    self.save_images(batch, e, i)
                    self.writer.add_scalar("Number_of_Nonzero_Entries", avg_mask, step)
                    self.writer.add_scalar("Training_Loss", training_loss, step)
                    #self.writer.add_scalar("Critic_Loss", critic_loss, step)
            #self.update_tau()
            #print(self.tau)    
            print("TESTING")
            for i, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                step = e*len(self.test_loader)+i
                testing_loss = self.testing_step(batch, e, i)
                if step % 50 == 0:
                    self.writer.add_scalar("Testing_Loss", testing_loss, step)
            print("SAVING")
            torch.save(self.network.state_dict(), self.result_dir + "/feature_network" + "_" + str(e) + ".pt")
            torch.save(self.FC.state_dict(), self.result_dir + "/FC_network" + "_" + str(e) + ".pt")
            torch.save(self.maskingnet.state_dict(), self.result_dir + "/mask_network" + "_" + str(e) + ".pt")
