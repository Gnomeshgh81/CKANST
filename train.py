import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

import KAN_StyTR
from module import basic_model, HSI
from module.ED import Encoder, Decoder
from utils.utils import InfiniteSamplerWrapper
from torchvision.utils import save_image


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'



parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='./datasets/train2014', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='./datasets/Images', type=str,  #wikiart dataset crawled from https://www.wikiart.org/
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')  #run the train.py, please download the pretrained vgg checkpoint

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--Train', type=bool, default=True)

parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--hidden_dim', default=128, type=int)

args = parser.parse_args()

# training setting
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

writer = SummaryWriter(log_dir=args.log_dir)
# model define
vgg = basic_model.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

encoder_c = Encoder(3, args.hidden_dim, 1)
encoder_c.load_state_dict(torch.load('./train-process/009/encoder_c_iter_70000.pth'))

encoder_s = Encoder(3, args.hidden_dim, 1)
encoder_s.load_state_dict(torch.load('./train-process/009/encoder_s_iter_70000.pth'))

KANdecoder = Decoder(args.hidden_dim, 1)
KANdecoder.load_state_dict(torch.load('./train-process/009/Kan_decoder_iter_70000.pth'))

HSI = HSI.HSI(args.hidden_dim)
HSI.load_state_dict(torch.load('./train-process/011/HSI_iter_160000.pth'))

decoder = basic_model.decoder
decoder.load_state_dict(torch.load("./train-process/009/decoder_iter_70000.pth"))


with torch.no_grad():
    network = KAN_StyTR.KanST(vgg, encoder_c, encoder_s, KANdecoder, decoder, HSI, args)

network.train()
network.to(device)
network = nn.DataParallel(network, device_ids=[0])


content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))

style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))

if not os.path.exists(args.save_dir+"/test"):
    os.makedirs(args.save_dir+"/test")


for i in tqdm(range(args.max_iter)):

    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    out, Icc, Iss, loss_c, loss_s, loss_lambda1, loss_lambda2  = network(content_images, style_images, i)

    if i % 100 == 0:
        output_name = os.path.join(args.save_dir, 'test', str(i).zfill(7) + '.jpg')
        print(output_name)
        out = torch.cat((content_images, out), 0)
        out = torch.cat((style_images, out), 0)

        output_name1 = os.path.join(args.save_dir, 'test', str(i).zfill(7) + '_Icc_.jpg')
        output_name2 = os.path.join(args.save_dir, 'test', str(i).zfill(7) + '_Iss_.jpg')

        save_image(out, output_name)
        save_image(Icc, output_name1)
        save_image(Iss, output_name2)

    loss = loss_c + loss_s

    print(loss.sum().cpu().detach().numpy(), "-content:", loss_c.sum().cpu().detach().numpy(), "-style:",
          loss_s.sum().cpu().detach().numpy(), "-loss_lambda1:", loss_lambda1.sum().cpu().detach().numpy(), "-loss_lambda2:",
          loss_lambda2.sum().cpu().detach().numpy())

    writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
    writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
    writer.add_scalar('loss_lambda1', loss_lambda1.sum().item(), i + 1)
    writer.add_scalar('loss_lambda2', loss_lambda2.sum().item(), i + 1)
    writer.add_scalar('total_loss', loss.sum().item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.module.encoder_c.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/encoder_c_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

        state_dict = network.module.encoder_s.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/encoder_s_iter_{:d}.pth'.format(args.save_dir,
                                                       i + 1))

        state_dict = network.module.Kan_decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/Kan_decoder_iter_{:d}.pth'.format(args.save_dir,
                                                         i + 1))

        state_dict = network.module.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                       i + 1))

        state_dict = network.module.HSI.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/HSI_iter_{:d}.pth'.format(args.save_dir,
                                                       i + 1))

writer.close()
