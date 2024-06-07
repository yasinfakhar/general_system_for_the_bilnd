import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from modules import EfficientFace
from config import Config
from PIL import Image
import time

label = ('Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger')
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
   
transforms_com = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        normalize])

softmax = nn.Softmax(dim=1)

def _load_model():
    
    model = EfficientFace.efficient_face()
    model.fc = nn.Linear(1024, 7)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(Config.facial_expression_model_path)
    pre_trained_dict = checkpoint['state_dict']
    model.load_state_dict(pre_trained_dict)
    model.eval()
    
    return model

model = _load_model()
print('[SUCCESS] Facial Expression Model loaded successfully')

def _preporcess(img , face):
    
    x0, y0 , x1, y1 = face['bbox']
    face_cropped = img[int(y0)-10:int(y1)+10, int(x0)-10:int(x1)+10, :]
    
    pil_image = Image.fromarray(face_cropped)
    face_cropped = transforms_com(pil_image)
    face_cropped = face_cropped.unsqueeze(0)
    face_cropped = face_cropped.cuda()
    
    return face_cropped
def recognize_emotion(img , face):

    processed_face = _preporcess(img , face)

    output = model(processed_face)
    output = softmax(output)

    return label[torch.argmax(output).item()]
