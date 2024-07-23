import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torchvision.utils import save_image

from PIL import Image
from IPython.display import Image as display_image

import cv2 as cv
import sys
import numpy as np

def calc_mean_std(feat, eps = 1e-5):
  size = feat.size()
  assert (len(size) == 4)
  N, C = size[:2]
  feat_var = feat.view(N, C, -1).var(dim = 2) + eps
  feat_std = feat_var.sqrt().view(N, C, 1, 1)
  feat_mean = feat.view(N, C, -1).mean(dim = 2).view(N, C, 1, 1)
  return feat_mean, feat_std

def adain(content_feat, style_feat):
  assert (content_feat.size()[:2] == style_feat.size()[:2])
  size = content_feat.size()
  style_mean, style_std = calc_mean_std(style_feat)
  content_mean, content_std = calc_mean_std(content_feat)

  normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
  return normalized_feat * style_std.expand(size) + style_mean.expand(size)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(), # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(), # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(), # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(), # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(), # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(), # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(), # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(), # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(), # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(), # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(), # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(), # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(), # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(), # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(), # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU() # relu5-4
)

decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4]) # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11]) # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18]) # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31]) # relu3_1 -> relu4_1
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1 
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat
        g_t = self.decoder(t) 
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return loss_c, loss_s

decoder.eval()
vgg.eval()

vgg_path = 'vgg_normalised.pth'
decoder_path = 'decoder.pth'

decoder.load_state_dict(torch.load(decoder_path))
vgg.load_state_dict(torch.load(vgg_path))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg.to(device)
decoder.to(device)

vgg = nn.Sequential(*list(vgg.children())[:31])   


    
def style_transfer(vgg, decoder, content, style, alpha = 1.0):
  assert (0.0 <= alpha <= 1.0)
  content_f = vgg(content)
  style_f = vgg(style)
  feat = adain(content_f, style_f)
  feat = feat * alpha + content_f * (1 - alpha)
  return decoder(feat)

def test_transform(size = 512):
  transform_list = []
  if size != 0:
    transform_list.append(transforms.Resize(size))
  transform_list.append(transforms.ToTensor())
  transform = transforms.Compose(transform_list)
  return transform

content_tf = test_transform()
style_tf = test_transform()

def OpenCV2PIL(opencv_image):
    color_coverted = cv.cvtColor(opencv_image, cv.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    return pil_image

def PIL2OpenCV(pil_image):
    numpy_image= np.array(pil_image)
    opencv_image = cv.cvtColor(numpy_image, cv.COLOR_RGB2BGR)
    return opencv_image

unloader = transforms.ToPILImage()

style_path = 'style_2.jpg'
display_image(style_path)

video_path = "content_3.mp4"
output_path = "C:/frames/output.mp4" 

cap = cv.VideoCapture(video_path)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'mp4v')

cnt = 0


fps = cap.get(cv.CAP_PROP_FPS)
out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

style = style_tf(Image.open(str(style_path)))
style = style.to(device).unsqueeze(0)

if not out.isOpened():
  sys.exit(f'Unable to open the video: {output_path}')

while cap.isOpened():
  ret, frame = cap.read()
  if ret:
    content = OpenCV2PIL(frame)
    content = content_tf(content)
    content = content.to(device).unsqueeze(0)
    output = style_transfer(vgg, decoder, content, style, alpha = 1.0)
    output = output.squeeze(0)
    output = PIL2OpenCV(unloader(output))
    output = cv.resize(output, (width, height), cv.INTER_LINEAR)
    out.write(output)
    print(cnt)
    cnt += 1
  else:
    break

out.release()
cap.release()
cv.destroyAllWindows()

print(f'Video Saved: {output_path}')