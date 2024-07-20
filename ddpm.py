import os
import logging
import torch
import torch.nn as nn
from torch import optim
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils import *
from modules import UNet
# from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO,datefmt="%I:%H:%S")

class Diffusion:
    def __init__(self,noise_steps=1000,beta_start=1e-4,beta_end=0.02,img_size=256,device="cpu"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)

        
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start,self.beta_end,self.noise_steps)
        
    def noise_images(self,x,t):
        #sqrt(a)x0 - sqrt(1-a)e
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:,None,None,None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:,None,None,None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e
    
    def sample_timesteps(self,n):
        return torch.randint(low=1,high=self.noise_steps,size=(n,))
    
    def sample(self,model,n):
        logging.info(f"Sampleing {n} new images.....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n,3,self.img_size,self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1,self.noise_steps)),position=0):
                t = (torch.ones(n)*i).long().to(self.device)
                predicted_noise = model(x,t)
                alpha = self.alpha[t][:,None,None,None]
                alpha_hat = self.alpha_hat[t][:,None,None,None]
                beta = self.beta[t][:,None,None,None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1,1)+1)/2
        x = (x*255).type(torch.uint8)
        return x

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataset,dataloader = get_data(args)
    model = UNet().to(device)
    optimizer = optim.AdamW(model.parameters(),lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size,device=device)
    # logger = SummaryWriter(os.path.join("runs",args.run_name))

    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)
        for i,(images,_) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t,noise = diffusion.noise_images(images,t)
            predicted_noise = model(x_t,t)
            loss = mse(noise,predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            sampled_images = diffusion.sample(model,n=images.shape[0])
            save_images(sampled_images,os.path.join("results",args.run_name,f"{epoch}.jpg"))
            torch.save(model.state_dict(), os.path.join("models",args.run_name,f"ckpt.pt"))

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Unconditional"
    args.epochs = 500
    args.batch_size = 12
    args.image_size = 64
    args.dataset_path = r"C:\Users\dome\datasets\landscape_img_folder"
    args.device = "cpu"
    args.lr = 3e-4
    train(args)

if __name__ == '__main__':
    launch()