import torch
from torch import optim
import torchvision.utils as vutils

from trainers.base_trainer import BaseTrainer
from models.networks import *

class VanillaTrainer(BaseTrainer):

    def __init__(self, params):
        super(VanillaTrainer, self).__init__(params)
        
        #Set device for training
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        #Build model
        self.network = Simple().to(self.device)
        self.FC = FC().to(self.device)
        self.maskingnet = MaskingNet().to(self.device)

        #Setup optimizer 
        pred_parameters = list(self.network.parameters()) + list(self.FC.parameters()) + list(self.maskingnet.parameters())
        self.pred_optimizer = optim.Adam(pred_parameters)
        
        #Define Losses
        self.loss_fn = nn.MSELoss()
        self.regularizer = nn.SmoothL1Loss()
        
    def training_step(self, batch, e, i):
        #Zero out gradients
        self.pred_optimizer.zero_grad()
    
        #Set up batch data
        images, segs = batch
        images = images.to(self.device)
        segs = segs.to(self.device)
        
        #Run through neural networks
        mask = self.maskingnet(images)[:, :, :, 1].unsqueeze(1)
        sparse_segs = segs * mask
        feats = self.network(images, sparse_segs)
        pred_segs = self.FC(feats)
        
        #Calculate loss 
        pred_loss = self.loss_fn(pred_segs, segs) 
        total_loss = pred_loss + self.regularizer(mask, torch.zeros_like(mask))
        
        #Optimize loss function
        total_loss.backward()
        self.pred_optimizer.step()
            
        return pred_loss.detach().cpu().numpy(), mask.data.nonzero().size(0)
        
    def testing_step(self, batch, e, i):
        with torch.no_grad():
            #Set up batch data
            images, segs = batch
            images = images.to(self.device)
            segs = segs.to(self.device)
        
            #Run through neural networks
            mask = self.maskingnet(images)[:, :, :, 1].unsqueeze(1)
            sparse_segs = segs * mask
            feats = self.network(images, sparse_segs)
            pred_segs = self.FC(feats)
        
            #Calculate loss
            loss = self.loss_fn(pred_segs, segs)
        return loss.detach().cpu().numpy()
    
    def save_models(self):
        torch.save(self.network.state_dict(), "cnn.pt")
        torch.save(self.FC.state_dict(), "fc.pt")
        torch.save(self.maskingnet.state_dict(), "maskingnet.pt")
        
    def save_images(self, batch, e, i):
        with torch.no_grad():
            images, segs = batch
            images = images.to(self.device)
            segs = segs.to(self.device)
            mask = self.maskingnet(images)[:, :, :, 1].unsqueeze(1)
            sparse_segs = segs * mask
            feats = self.network(images, sparse_segs)
            pred_segs = self.FC(feats)
       
        vutils.save_image(images, self.result_dir + "/" + str(i) + "_training_images_" + str(e) + ".png", normalize=True)
        vutils.save_image(mask, self.result_dir + "/" + str(i) + "_training_masks_" + str(e) + ".png", normalize=True)
        #vutils.save_image(raw_mask[:, :, :, 1].view(-1, 1, 100, 100), self.result_dir + "/" + str(i) + "_training_raw_masks_" + str(e) + ".png", normalize=True)
        vutils.save_image(segs, self.result_dir + "/" + str(i) + "_training_segs_" + str(e) + ".png", normalize=True)
        vutils.save_image(pred_segs, self.result_dir + "/" + str(i) + "_predicted_segs_" + str(e) + ".png", normalize=True)
