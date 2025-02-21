import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import zipfile
import PIL.Image
import json
import dnnlib
import pickle
from tqdm import tqdm
try:
    import pyspng
except ImportError:
    pyspng = None
import matplotlib.pyplot as plt
import argparse

def load_img_zip(zip_path):
    zip_file = zipfile.ZipFile(zip_path)
    all_names = set(zip_file.namelist())

    PIL.Image.init()
    image_names = sorted(fname for fname in all_names if file_ext(fname) in PIL.Image.EXTENSION)

    images = []    
    # load images
    for name in tqdm(image_names):
        with zip_file.open(name, 'r') as f:
            if pyspng is not None and file_ext(name) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)     # HWC => CHW

        # append images
        images.append(image[np.newaxis, :, :, :])

    images = np.concatenate(images, axis=0)
    N, C, H, W = images.shape
    #y = torch.from_numpy(images).to(torch.float32) / 127.5 - 1
    y = images / 127.5 - 1
    y = y.reshape(N, -1)
    y = np.float32(y)
    return y

def file_ext(fname):
    return os.path.splitext(fname)[1].lower()

def load_ffhq_zip(zip_path):
    zip_file = zipfile.ZipFile(zip_path)
    all_names = set(zip_file.namelist())

    PIL.Image.init()
    image_names = sorted(fname for fname in all_names if file_ext(fname) in PIL.Image.EXTENSION)

    images = []    
    # load images
    for name in tqdm(image_names):
        with zip_file.open(name, 'r') as f:
            if pyspng is not None and file_ext(name) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)     # HWC => CHW

        # append images
        images.append(image[np.newaxis, :, :, :])

    images = np.concatenate(images, axis=0)
    N, C, H, W = images.shape
    y = torch.from_numpy(images).to(torch.float32) / 127.5 - 1
    y = y.reshape(N, C*H*W)
    return y

def normalize_img(img):
    img = img-torch.min(img)
    img = img/torch.max(img)
    return img

def get_denoised_img(noisy_img, net, var, device=torch.device('cuda')):
    # variance should be in the range(0.002,80)
    return net(noisy_img, torch.tensor(var,device=device))

# This dataset contains only images
class Image_Only_Dataset(Dataset):
    def __init__(self, data, device, max_num_images=1000):
        # data is the noisy image, target is the denoised image
        max_num_images = min(len(data), max_num_images)
        self.data = torch.from_numpy(data[:max_num_images,:]).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def get_dataloader(data_path, batch_size):
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data_path, use_labels=False, xflip=False, cache=True)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2)
    dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs) # subclass of training.dataset.Dataset
    dataloader = torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=batch_size, shuffle=False)
    return dataset_obj

class LinearModel(nn.Module):
    def __init__(self, input_dim, sigma, temperature=1):
        super(LinearModel, self).__init__()
        self.linear_mapping = nn.Linear(input_dim, input_dim)
        nn.init.constant_(self.linear_mapping.bias, 0)
        nn.init.constant_(self.linear_mapping.weight, 0)
        self.sigma =  sigma
        self.temp = temperature
        self.input_dim = input_dim

    def forward(self, queries):
        queries = queries/(self.temp*self.input_dim ** 0.5)
        output = self.linear_mapping(queries)
        return output
    
def noisy_image_distllation(sigma, lr, temperature, epochs, network_pkl_final, image_path, input_dim):
    '''
    Sigma: the noise variance, we consider sigma in [80.0, 42.415, 21.108, 9.723, 4.06, 1.501, 0.469, 0.116, 0.020, 0.002]
    lr: learning rate for training the linear model. Importantly, different noise levels require differen learning rate to train correctly !!!
    temperature: we fix this to 1
    epochs: number of epochs for training
    network_pkl_final: the nonlinear diffusion model we want to approximate
    image path: image dataset used for linear estimation; For diffusion models traind on FFHQ, we perform linear estimation on Celeb-hq dataset. For diffusion models trained on Cifar-10, we perform linear estimation on Cifar-10 test set.
    input dim: image dimension, 3*64*64 for FFHQ, 3*32*32 for Cifar-10
    '''
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cuda')
    
    max_num_images = 1e9
    # Constructs dataset
    batch_size = 256
    
    # Construct attention layer
    sigma = sigma
    temperature = temperature
    model = LinearModel(input_dim, sigma, temperature=temperature).to(device)
    
    # Load the target network
    print(f'Loading network from "{network_pkl_final}"...')
    with dnnlib.util.open_url(network_pkl_final) as f:
        net = pickle.load(f)['ema'].to(device)
        
    # Set hyperparameters for traning
    lr = lr # 5e-5, 5e-4, 5e-3, 5e-2, 5e-1
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print_every = 1e3
    save_every = 5
    epochs = epochs

    # Prepare image dataset for training
    data_train = load_img_zip(image_path)
    dataset_train = Image_Only_Dataset(data_train, device, max_num_images)
    data_loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)

    # Begin training
    save_dir = os.path.join('Linear',str(round(sigma,4)),'noisy_image_distillation')
    total_loss_list = []
    for epoch in range(epochs):
        # Training
        total_loss_train = 0
        for data in data_loader_train:
            noisy_image = data + sigma*torch.randn_like(data)
            denoised_pred = model(noisy_image)
            denoised_gt = get_denoised_img(noisy_image.reshape(noisy_image.shape[0],3,64,64), net, sigma).reshape(noisy_image.shape[0],-1)
            loss = criterion(denoised_pred, denoised_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()
        total_loss_train = total_loss_train/len(data_loader_train)
        total_loss_list.append(total_loss_train)

        if epoch % print_every == 0 or epoch == epochs-1:
            print(f"Epoch [{epoch+1}/{epochs}], train Loss: {total_loss_train:.4f}")
            plt.figure()
            plt.plot(total_loss_list)
            plt.show()
        if epoch % save_every == 0 or epoch == epochs-1:
            if os.path.exists(save_dir+'/'+str(lr)) == False:
                os.makedirs(save_dir+'/'+str(lr))
            torch.save(model.state_dict(), os.path.join(save_dir, str(lr), 'weights'+str(epoch)+'.pt'))
            torch.save(total_loss_list, os.path.join(save_dir, str(lr), 'train_loss_list.pt'))
    return total_loss_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train_linear')
    parser.add_argument('--sigma', type=float, required=True, help='Value of sigma')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate for trianing the model')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--network_pkl_final', type=str, required=True, help='network weights')
    parser.add_argument('--image_path', type=str, required=True, help='path to the image')
    parser.add_argument('--input_dim', type=str, required=True, help='dimension of the image, i.e., C*H*W')

    args = parser.parse_args()
    noisy_image_distllation(args.sigma, args.lr, args.epochs, args.network_pkl_final, args.image_path, args.input_dim)


