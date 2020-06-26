import torch
from torch import optim
import torchvision.utils as vutils

from trainers.base_trainer import BaseTrainer
from models.networks import *
from models.gumbel_softmax import *

class DDPGTrainer(BaseTrainer):

    def __init__(self, params):
        super(DDPGTrainer, self).__init__(params)
        
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
        self.pred_critic = CriticNet().to(self.device)
        self.mask_critic = CriticNet().to(self.device)

        #Setup optimizer
        mask_parameters = list(self.maskingnet.parameters()) 
        pred_parameters = list(self.network.parameters()) + list(self.FC.parameters())
        #critic_parameters = list(self.pred_critic.parameters()) + list(self.mask_critic.parameters())
        self.mask_optimizer = optim.Adam(mask_parameters, 3e-4)
        self.pred_optimizer = optim.Adam(pred_parameters, 3e-4)
        self.mcritic_optimizer = optim.Adam(self.mask_critic.parameters(), 3e-2)
        self.pcritic_optimizer = optim.Adam(self.pred_critic.parameters(), 3e-2)
        
    def training_step(self, batch, e, i):
        #Zero out gradients
        self.mask_optimizer.zero_grad()
        self.pred_optimizer.zero_grad()
        self.mcritic_optimizer.zero_grad()
        self.pcritic_optimizer.zero_grad()
        
        lmbda = 1000*e
        if i == 0:
            print(lmbda)
        
        #Set up batch data
        images, segs = batch
        images = images.to(self.device)
        segs = segs.to(self.device)
        
        #Run through neural networks
        raw_mask = self.maskingnet(images)
        mask = generate_mask(raw_mask)
        raw_mask = generate_mask(raw_mask, noisy=False)
        sparse_segs = mask * segs
        feats = self.network(images, sparse_segs)
        pred_segs = self.FC(feats)
        raw_values = self.pred_critic(images, raw_mask) + lmbda*self.mask_critic(images, raw_mask)
        mask_values = self.pred_critic(images, mask.detach()) + lmbda*self.mask_critic(images, mask.detach())
        
        #Calculate loss 
        pred_loss = torch.mean((pred_segs - segs)**2, dim=[2, 3])
        #total_loss = (pred_loss + lmbda*mask.abs().mean(dim=[2, 3])).detach()
        mcritic_loss = ((self.mask_critic(images, mask.detach()) - mask.abs().mean(dim=[2, 3]).detach())**2).mean()
        pcritic_loss = ((self.pred_critic(images, mask.detach()) - pred_loss.detach())**2).mean()
        #critic_loss = ((self.pred_critic(images, mask.detach()) - pred_loss.detach())**2 + (self.mask_critic(images, mask.detach()) - mask.abs().mean(dim=[2, 3]).detach())**2).mean()
        critic_loss = mcritic_loss + pcritic_loss
        mask_loss = torch.mean(raw_values)
        pred_loss = pred_loss.mean()
        
        #Optimize loss functions
        pred_loss.backward(retain_graph=True)
        self.pred_optimizer.step()
        mask_loss.backward()
        self.mask_optimizer.step()
        mcritic_loss.backward()
        pcritic_loss.backward()
        #critic_loss.backward()
        self.mcritic_optimizer.step()
        self.pcritic_optimizer.step()
        
        return pred_loss.detach().cpu().numpy(), mask.data.nonzero().size(0), critic_loss.detach().cpu().numpy()
        
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
            raw_mask = self.maskingnet(images)
            mask = generate_mask(raw_mask)
            sparse_segs = mask * segs
            feats = self.network(images, sparse_segs)
            pred_segs = self.FC(feats)
        
        vutils.save_image(mask, self.result_dir + "/" + str(i) + "_training_masks_" + str(e) + ".png", normalize=True)
        vutils.save_image(raw_mask[:, :, :, 1].view(-1, 1, 100, 100), self.result_dir + "/" + str(i) + "_training_raw_masks_" + str(e) + ".png", normalize=True)
        vutils.save_image(segs, self.result_dir + "/" + str(i) + "_training_segs_" + str(e) + ".png", normalize=True)
        vutils.save_image(pred_segs, self.result_dir + "/" + str(i) + "_predicted_segs_" + str(e) + ".png", normalize=True)
