import os
import yaml
import datetime
import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

class BaseTrainer(object):

    def __init__(self, params):

        self.params = params
        
        input_trans = transforms.Compose([transforms.CenterCrop(300), transforms.Resize(100), transforms.ToTensor()])
        target_trans = transforms.Compose([transforms.CenterCrop(300),  transforms.Resize(100), transforms.ToTensor()])
        
        train_data = torchvision.datasets.Cityscapes('data', split="train", target_type="semantic", mode="fine", transform=input_trans, target_transform=target_trans)
        test_data = torchvision.datasets.Cityscapes('data', split="test", target_type="semantic", mode="fine", transform=input_trans, target_transform=target_trans)
        
        self.train_loader = torch.utils.data.DataLoader(train_data,batch_size=self.params["batch_size"])
        self.test_loader = torch.utils.data.DataLoader(test_data,batch_size=self.params["batch_size"])
        
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
            
            #Training Loop
            print("TRAINING")
            for i, batch in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
                self.training_step(batch, e, i)
                
                if i in [1, 5, 10, 100]:
                    self.save_images(batch, e, i)
                
            print("TESTING")
            epoch_loss = 0.0
            for i, batch in tqdm(enumerate(self.test_loader), total=len(self.test_loader)):
                epoch_loss += self.testing_step(batch, e, i)
            print(epoch_loss / len(self.test_loader))
