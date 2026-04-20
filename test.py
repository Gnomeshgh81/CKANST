import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image

import numpy as np

import KAN_StyTR
from module import basic_model
from module.ED import Encoder, Decoder
from module.HSI import HSI


def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transform(h, w):
    k = (h, w)
    size = int(np.max(k))
    print(type(size))
    transform_list = []
    transform_list.append(transforms.CenterCrop((h, w)))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def content_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content', type=str, help='File path to the content image')
parser.add_argument('--content_dir', type=str, help='Directory path to a batch of content images')
parser.add_argument('--style', type=str, help='File path to the style image')
parser.add_argument('--style_dir', type=str, help='Directory path to a batch of style images')
parser.add_argument('--output', type=str, default='output', help='Directory to save the output image(s)')
parser.add_argument('--Train', type=bool, default=False, help="Is the model in training mode?")
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')
parser.add_argument('--encoder_c', type=str, default='./train-process/011/encoder_c_iter_160000.pth')
parser.add_argument('--encoder_s', type=str, default='./train-process/011/encoder_s_iter_160000.pth')
parser.add_argument('--Kan_decoder', type=str, default='./train-process/011/Kan_decoder_iter_160000.pth')
parser.add_argument('--HSI', type=str, default='./train-process/011/HSI_iter_160000.pth')
parser.add_argument('--decoder', type=str, default="./train-process/011/decoder_iter_160000.pth")
parser.add_argument('--hidden_dim', type=int, default=128, help="The dimension of the hidden layers of the model")
args = parser.parse_args()

# Advanced options
content_size = 512
style_size = 512
crop = 'store_true'
save_ext = '.jpg'
output_path = args.output
preserve_color = 'store_true'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Either --content or --content_dir should be given.
if args.content:
    content_paths = [Path(args.content)]
else:
    content_dir = Path(args.content_dir)
    content_paths = [f for f in content_dir.glob('*')]

# Either --style or --style_dir should be given.
if args.style:
    style_paths = [Path(args.style)]
else:
    style_dir = Path(args.style_dir)
    style_paths = [f for f in style_dir.glob('*')]

if not os.path.exists(output_path):
    os.mkdir(output_path)

# model define
vgg = basic_model.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

encoder_c = Encoder(3, args.hidden_dim, 1)
encoder_c.load_state_dict(torch.load(args.encoder_c))

encoder_s = Encoder(3, args.hidden_dim, 1)
encoder_s.load_state_dict(torch.load(args.encoder_s))

KANdecoder = Decoder(args.hidden_dim, 1)
KANdecoder.load_state_dict(torch.load(args.Kan_decoder))

HSI = HSI(args.hidden_dim)
HSI.load_state_dict(torch.load(args.HSI))

decoder = basic_model.decoder
decoder.load_state_dict(torch.load(args.decoder))

encoder_c.eval()
encoder_s.eval()
KANdecoder.eval()
HSI.eval()
decoder.eval()
vgg.eval()

network = KAN_StyTR.KanST(vgg, encoder_c, encoder_s, KANdecoder, decoder, HSI, args)
network.eval()
network.to(device)

# data transform
content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)

# test
for content_path in content_paths:
    for style_path in style_paths:
        print(content_path)

        content_tf1 = content_transform()
        content = content_tf(Image.open(content_path).convert("RGB"))

        h, w, c = np.shape(content)
        style_tf1 = style_transform(h, w)
        style = style_tf(Image.open(style_path).convert("RGB"))

        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)

        with torch.no_grad():
            output = network(content, style, 1)

        output_img = output.cpu()

        output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
            output_path, splitext(basename(content_path))[0],
            splitext(basename(style_path))[0], save_ext
        )
        save_image(output_img, output_name)