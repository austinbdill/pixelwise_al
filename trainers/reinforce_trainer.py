import torch
from torch import optim
import torchvision.utils as vutils

from trainers.base_trainer import BaseTrainer
from models.networks import *

class ReinforceTrainer(BaseTrainer):

    def __init__(self, params):
        super(ReinforceTrainer, self).__init__(params)
        
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
        pred_parameters = list(self.network.parameters()) + list(self.FC.parameters())
        self.pred_optimizer = optim.Adam(pred_parameters)
        self.mask_optimizer = optim.Adam(self.maskingnet.parameters())
        
        self.lmbda = 0
       
        
    def training_step(self, batch, e, i):
        #Zero out gradients
        self.pred_optimizer.zero_grad()
        self.mask_optimizer.zero_grad()
        
        #This needs to be replaced with a separate class to be shared by all trainers
        self.lmbda = min(1.5*e, 60)
        if i == 0:
            print(self.lmbda)
    
        #Set up batch data
        images, segs = batch
        images = images.to(self.device)
        segs = segs.to(self.device)
        
        #Run through neural networks
        mask_probs = self.maskingnet(images)[:, :, :, 1].unsqueeze(1)
        bernoulli = torch.distributions.bernoulli.Bernoulli(probs=mask_probs)
        mask = bernoulli.sample()
        log_probs = bernoulli.log_prob(mask)
        sparse_segs = segs * mask
        feats = self.network(images, sparse_segs)
        pred_segs = self.FC(feats)
        
        #Calculate loss 
        pred_loss = (pred_segs - segs)**2 
        #total_loss = (pred_loss + lmbda*self.regularizer(mask, torch.zeros_like(mask))).detach()
        total_loss = (pred_loss + self.lmbda*mask.abs()).detach()
        mask_loss = torch.mean(log_probs*(total_loss-torch.mean(total_loss)))
        
        #Optimize loss functions
        pred_loss = pred_loss.mean()
        pred_loss.backward()
        self.pred_optimizer.step()
        mask_loss.backward()
        self.mask_optimizer.step()
            
        return pred_loss.detach().cpu().numpy(), mask.data.nonzero().size(0), 0
        
    def testing_step(self, batch, e, i):
        with torch.no_grad():
            #Set up batch data
            images, segs = batch
            images = images.to(self.device)
            segs = segs.to(self.device)
        
            #Run through neural networks
            mask_probs = self.maskingnet(images)[:, :, :, 1].unsqueeze(1)
            bernoulli = torch.distributions.bernoulli.Bernoulli(probs=mask_probs)
            mask = bernoulli.sample()
            sparse_segs = segs * mask
            feats = self.network(images, sparse_segs)
            pred_segs = self.FC(feats)
        
            #Calculate loss
            loss = torch.mean((pred_segs-segs)**2)
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
            mask_probs = self.maskingnet(images)[:, :, :, 1].unsqueeze(1)
            bernoulli = torch.distributions.bernoulli.Bernoulli(probs=mask_probs)
            mask = bernoulli.sample()
            sparse_segs = segs * mask
            feats = self.network(images, sparse_segs)
            pred_segs = self.FC(feats)
       
        vutils.save_image(images, self.result_dir + "/" + str(i) + "_training_images_" + str(e) + ".png", normalize=True)
        vutils.save_image(mask, self.result_dir + "/" + str(i) + "_training_masks_" + str(e) + ".png", normalize=True)
        #vutils.save_image(raw_mask[:, :, :, 1].view(-1, 1, 100, 100), self.result_dir + "/" + str(i) + "_training_raw_masks_" + str(e) + ".png", normalize=True)
        vutils.save_image(segs, self.result_dir + "/" + str(i) + "_training_segs_" + str(e) + ".png", normalize=True)
        vutils.save_image(pred_segs, self.result_dir + "/" + str(i) + "_predicted_segs_" + str(e) + ".png", normalize=True)
