import torch
from torch import optim
import torchvision.utils as vutils

from trainers.base_trainer import BaseTrainer
from models.networks import *
from models.gumbel_softmax import *

class GumbelTrainer(BaseTrainer):

    def __init__(self, params):
        super(GumbelTrainer, self).__init__(params)
        
        #Set device for training
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        #Build model
        self.network = Simple().to(self.device)
        self.FC = FC().to(self.device)
        self.maskingnet = MaskingNet().to(self.device)
        self.tau = 10

        #Setup optimizer
        self.parameters = list(self.network.parameters()) + list(self.FC.parameters()) + list(self.maskingnet.parameters())
        self.optimizer = optim.Adam(self.parameters)
        
        #Define Losses
        self.loss_fn = nn.MSELoss() 
        self.regularizer = nn.L1Loss()
        
    def update_tau(self):
        self.tau = max(self.tau*0.5, 0.01)
        
    def training_step(self, batch, e, i):
    
        # Swap this out for a new method
        lmbda = e
        if i == 0:
            print(lmbda)
    
        #Zero out gradients
        self.optimizer.zero_grad()
        
        #Set up batch data
        images, segs = batch
        images = images.to(self.device)
        segs = segs.to(self.device)
        
        #Run through neural networks
        raw_mask = self.maskingnet(images)
        mask = generate_mask(raw_mask, temp=self.tau, noisy=(i % 2 == 0))
        sparse_segs = mask * segs
        feats = self.network(images, sparse_segs)
        pred_segs = self.FC(feats)
        
        #Calculate loss
        pred_loss = torch.mean((pred_segs - segs)**2)
        mask_loss = mask.abs().sum()
        total_loss = pred_loss + lmbda*mask_loss
        
        #Optimize loss function
        total_loss.backward()
        self.optimizer.step()
        return pred_loss.detach().cpu().numpy(), mask.data.nonzero().size(0), 0
        
    def testing_step(self, batch, e, i):
        with torch.no_grad():
            #Set up batch data
            images, segs = batch
            images = images.to(self.device)
            segs = segs.to(self.device)
        
            #Run through neural networks
            raw_mask = self.maskingnet(images)
            mask = generate_mask(raw_mask, temp=self.tau, noisy=True)
            sparse_segs = mask * segs
            feats = self.network(images, sparse_segs)
            pred_segs = self.FC(feats)
        
            #Calculate loss
            loss = self.loss_fn(pred_segs, segs) #+ 100*self.regularizer(mask, torch.zeros_like(mask))
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
            mask = generate_mask(raw_mask, noisy=True, temp=self.tau)
            sparse_segs = mask * segs
            feats = self.network(images, sparse_segs)
            pred_segs = self.FC(feats)

        vutils.save_image(raw_mask[:, :, :, 0].unsqueeze(1), self.result_dir + "/" + str(i) + "_training_first_raw_" + str(e) + ".png", normalize=True)
        vutils.save_image(raw_mask[:, :, :, 1].unsqueeze(1), self.result_dir + "/" + str(i) + "_training_second_raw_" + str(e) + ".png", normalize=True)
        vutils.save_image(mask, self.result_dir + "/" + str(i) + "_training_masks_" + str(e) + ".png", normalize=True)
        vutils.save_image(segs, self.result_dir + "/" + str(i) + "_training_segs_" + str(e) + ".png", normalize=True)
        vutils.save_image(pred_segs, self.result_dir + "/" + str(i) + "_predicted_segs_" + str(e) + ".png", normalize=True)
