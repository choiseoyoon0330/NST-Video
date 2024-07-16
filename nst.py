import sys
import cv2 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)

imsize = (256, 256) 

loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

def image_loader(image):
  image = loader(image).unsqueeze(0)
  return image.to(device, torch.float)

def OpenCV2PIL(opencv_img):
    color_converted = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(color_converted)
    return pil_img

def PIL2OpenCV(pil_img):
    numpy_img = np.array(pil_img)
    opencv_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2BGR)
    return opencv_img

style_img = cv2.imread("C:\style_1.jpg")      # type: OpenCV

if style_img is None:
    sys.exit('파일을 찾을 수 없습니다')
    
cv2.imshow('Image Display', style_img)
cv2.waitKey(1000)
cv2.destroyAllWindows()

style_img = OpenCV2PIL(style_img)             # type: PILImage
style_img = image_loader(style_img)           # type: Tensor


unloader = transforms.ToPILImage()


class ContentLoss(nn.Module):

  def __init__(self, target):
    super(ContentLoss, self).__init__()
    self.target = target.detach()

  def forward(self, input):
    self.loss = F.mse_loss(input, self.target)
    return input

def gram_matrix(input):
  n, c, h, w = input.size()

  features = input.view(n * c, h * w)

  G = torch.mm(features, features.t())

  return G.div(n * c * h * w)

class StyleLoss(nn.Module):

  def __init__(self, target_feature):
    super(StyleLoss, self).__init__()
    self.target = gram_matrix(target_feature).detach()

  def forward(self, input):
    G = gram_matrix(input)
    self.loss = F.mse_loss(G, self.target)
    return input

cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

mean_rgb = torch.tensor([0.485, 0.456, 0.406])
std_rgb = torch.tensor([0.229, 0.224, 0.225])

class Normalization(nn.Module):

  def __init__(self, mean, std):
    super(Normalization, self).__init__()
    self.mean = torch.tensor(mean).view(-1, 1, 1)
    self.std = torch.tensor(std).view(-1, 1, 1)

  def forward(self, img):
    return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers = content_layers_default,
                               style_layers = style_layers_default):

  normalization = Normalization(normalization_mean, normalization_std)

  content_losses = []
  style_losses = []

  model = nn.Sequential(normalization)

  i = 0
  for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
      i += 1
      name = 'conv_{}'.format(i)
    elif isinstance(layer, nn.ReLU):
      name = 'relu_{}'.format(i)
      layer = nn.ReLU(inplace = False)
    elif isinstance(layer, nn.MaxPool2d):
      name = 'pool_{}'.format(i)
    elif isinstance(layer, nn.BatchNorm2d):
      name = 'bn_{}'.format(i)
    else:
      raise RuntimeError('Unrecognized Layer : {}'.format(layer.__class__.__name__))

    model.add_module(name, layer)

    if name in content_layers:
      target = model(content_img).detach()
      content_loss = ContentLoss(target)
      model.add_module('content_loss_{}'.format(i), content_loss)
      content_losses.append(content_loss)

    if name in style_layers:
      target_feature = model(style_img).detach()
      style_loss = StyleLoss(target_feature)
      model.add_module('style_loss_{}'.format(i), style_loss)
      style_losses.append(style_loss)

  for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
      break

  model = model[:(i + 1)]

  return model, style_losses, content_losses

def get_input_optimizer(input_img):
  optimizer = optim.LBFGS([input_img])
  return optimizer

def style_transfer(cnn, normalization_mean, normalization_std,
                   content_img, style_img, input_img, num_steps = 150,
                   style_weight = 1000000, content_weight = 1):

  model, style_losses, content_losses = get_style_model_and_losses(cnn,
      normalization_mean, normalization_std, style_img, content_img)

  input_img.requires_grad_(True)

  model.eval()
  model.requires_grad_(False)

  optimizer = get_input_optimizer(input_img)


  run = [0]
  while run[0] <= num_steps:

    def closure():
      with torch.no_grad():
        input_img.clamp_(0, 1)

      optimizer.zero_grad()
      model(input_img)
      style_score = 0
      content_score = 0

      for sl in style_losses:
        style_score += sl.loss
      for cl in content_losses:
        content_score += cl.loss

      style_score *= style_weight
      content_score *= content_weight

      loss = style_score + content_score
      loss.backward()
      run[0] += 1

      return style_score + content_score

    optimizer.step(closure)
  with torch.no_grad():
    input_img.clamp_(0, 1)

  return input_img


filepath = "C:/content_3.mp4"

cap = cv2.VideoCapture(filepath)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_path = "C:/frames/output.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

if not out.isOpened():
    sys.exit(f'Unable to open the video: {output_path}')
    
cnt = 0
while cap.isOpened():
  ret, frame = cap.read()
  if ret:
    if cnt % 5 == 0:
      print(cnt)  
      
      content_img = OpenCV2PIL(frame)      # type: PILImage
      content_img = image_loader(content_img)   
      input_img = content_img.clone()
      output = style_transfer(cnn, mean_rgb, std_rgb,
                        content_img, style_img, input_img)
      output = output.squeeze(0)
      output = unloader(output)
      output = PIL2OpenCV(output)
      output = cv2.resize(output, (width, height))
      out.write(output)

    cnt += 1
  else:
    break

out.release()
cap.release()
cv2.destroyAllWindows()

print(f'Video Saved: {output_path}')

