import torch
from torch import optim
import torchvision.utils as vutils

from trainers.base_trainer import BaseTrainer
from models.networks import *

class A2CTrainer(BaseTrainer):

    def __init__(self, params):
        super(A2CTrainer, self).__init__(params)
        
        #Set device for training
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        #Build model
        self.network = Simple().to(self.device)
        self.FC = FC().to(self.device)
        self.maskingnet = MaskingNet().to(self.device)
        self.critic = CriticNet().to(self.device)

        #Setup optimizer 
        pred_parameters = list(self.network.parameters()) + list(self.FC.parameters())
        self.pred_optimizer = optim.Adam(pred_parameters, lr=3e-4)
        self.mask_optimizer = optim.Adam(self.maskingnet.parameters(), 3e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), 3e-4)
        
    def training_step(self, batch, e, i):
        #Zero out gradients
        self.pred_optimizer.zero_grad()
        lmbda = 200*e
        if i == 0:
            print(lmbda)
    
        #Set up batch data
        images, segs = batch
        images = images.to(self.device)
        segs = segs.to(self.device)
        
        #Run through neural networks
        mask_probs = self.maskingnet(images)[:, :, :, 1].unsqueeze(1)
        bernoulli = torch.distributions.bernoulli.Bernoulli(probs=mask_probs)
        mask = bernoulli.sample()
        log_probs = bernoulli.log_prob(mask).sum(dim=[2, 3])
        sparse_segs = segs * mask
        feats = self.network(images, sparse_segs)
        pred_segs = self.FC(feats)
        values = self.critic(images, mask.detach())
        
        #Calculate loss 
        pred_loss = torch.mean((pred_segs - segs)**2, dim=[2, 3])
        total_loss = (pred_loss + lmbda*mask.abs().mean(dim=[2, 3])).detach()
        critic_loss = ((values - total_loss)**2).mean()
        mask_loss = torch.mean(log_probs*(total_loss-values.detach()))
        
        pred_loss = pred_loss.mean()
        
        #Optimize loss functions
        pred_loss.backward()
        self.pred_optimizer.step()
        mask_loss.backward()
        self.mask_optimizer.step()
        critic_loss.backward()
        self.critic_optimizer.step()
            
        return pred_loss.detach().cpu().numpy(), mask.data.nonzero().size(0)
        
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
            loss = torch.mean((pred_segs - segs)**2)
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
            print(mask_probs.min())
            sparse_segs = segs * mask
            feats = self.network(images, sparse_segs)
            pred_segs = self.FC(feats)
       
        vutils.save_image(images, self.result_dir + "/" + str(i) + "_training_images_" + str(e) + ".png", normalize=True)
        vutils.save_image(mask, self.result_dir + "/" + str(i) + "_training_masks_" + str(e) + ".png", normalize=True)
        #vutils.save_image(raw_mask[:, :, :, 1].view(-1, 1, 100, 100), self.result_dir + "/" + str(i) + "_training_raw_masks_" + str(e) + ".png", normalize=True)
        vutils.save_image(segs, self.result_dir + "/" + str(i) + "_training_segs_" + str(e) + ".png", normalize=True)
        vutils.save_image(pred_segs, self.result_dir + "/" + str(i) + "_predicted_segs_" + str(e) + ".png", normalize=True)
