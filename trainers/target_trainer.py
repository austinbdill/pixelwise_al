import torch
from torch import optim
import torchvision.utils as vutils

from trainers.base_trainer import BaseTrainer
from models.networks import *
from models.gumbel_softmax import *

class TargetTrainer(BaseTrainer):

    def __init__(self, params):
        super(TargetTrainer, self).__init__(params)
        
        #Set device for training
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        self.tau = self.params["tau"]
        
        #Build model
        self.network = Simple().to(self.device)
        self.FC = FC().to(self.device)
        self.maskingnet = MaskingNet().to(self.device)
        
        self.target_network = Simple().to(self.device)
        self.target_FC = FC().to(self.device)
        self.target_maskingnet = MaskingNet().to(self.device)
        
        # We initialize the target networks as copies of the original networks
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_FC.parameters(), self.FC.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_maskingnet.parameters(), self.maskingnet.parameters()):
            target_param.data.copy_(param.data)

        #Setup optimizer
        mask_parameters = list(self.maskingnet.parameters()) 
        pred_parameters = list(self.network.parameters()) + list(self.FC.parameters())
        self.mask_optimizer = optim.Adam(mask_parameters)
        self.pred_optimizer = optim.Adam(pred_parameters)
        
        #Define Losses
        self.loss_fn = nn.MSELoss() 
        self.regularizer = nn.SmoothL1Loss()
        
    def training_step(self, batch, e, i):
        #Zero out gradients
        self.mask_optimizer.zero_grad()
        
        #Set up batch data
        images, segs = batch
        images = images.to(self.device)
        segs = segs.to(self.device)
        
        #Call weighting method
        
        #Run through neural networks
        raw_mask = self.maskingnet(images)
        mask = generate_mask(raw_mask)
        sparse_segs = mask * segs
        feats = self.target_network(images, sparse_segs)
        pred_segs = self.target_FC(feats)
        
        #Calculate loss
        mask_loss = self.loss_fn(pred_segs, segs) + self.regularizer(mask, torch.zeros_like(mask))
        
        #Optimize loss function
        mask_loss.backward()
        self.mask_optimizer.step()
        
        #Zero out gradients
        self.pred_optimizer.zero_grad()
        
        #Run through neural networks
        raw_mask = self.target_maskingnet(images)
        mask = generate_mask(raw_mask, temp=0.1).detach()
        sparse_segs = mask * segs
        feats = self.network(images, sparse_segs)
        pred_segs = self.FC(feats)
        
        #Calculate loss
        pred_loss = self.loss_fn(pred_segs, segs) + self.regularizer(mask, torch.zeros_like(mask))
        
        #Optimize loss function
        pred_loss.backward()
        self.pred_optimizer.step()
        
        #Update target weights
        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_FC.parameters(), self.FC.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_maskingnet.parameters(), self.maskingnet.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
            
        return pred_loss.detach().cpu().numpy(), mask.data.nonzero().size(0), 0
        
    def testing_step(self, batch, e, i):
        with torch.no_grad():
            #Set up batch data
            images, segs = batch
            images = images.to(self.device)
            segs = segs.to(self.device)
        
            #Run through neural networks
            raw_mask = self.maskingnet(images)
            mask = generate_mask(raw_mask)
            sparse_segs = mask * segs
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
            raw_mask = self.maskingnet(images)
            mask = generate_mask(raw_mask)
            sparse_segs = mask * segs
            feats = self.network(images, sparse_segs)
            pred_segs = self.FC(feats)
        
        vutils.save_image(mask, self.result_dir + "/" + str(i) + "_training_masks_" + str(e) + ".png", normalize=True)
        vutils.save_image(raw_mask[:, :, :, 1].view(-1, 1, 100, 100), self.result_dir + "/" + str(i) + "_training_raw_masks_" + str(e) + ".png", normalize=True)
        vutils.save_image(segs, self.result_dir + "/" + str(i) + "_training_segs_" + str(e) + ".png", normalize=True)
        vutils.save_image(pred_segs, self.result_dir + "/" + str(i) + "_predicted_segs_" + str(e) + ".png", normalize=True)
