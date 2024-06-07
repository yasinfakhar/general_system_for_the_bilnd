import torch
from modules.clipseg import CLIPDensePredT
from torchvision import transforms
from config import Config

transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
transforms.Resize((352, 352)),
])

def _load_model():
    
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True).cuda()
    model.eval()
    model.load_state_dict(torch.load(Config.finding_objects_model_path,
                                     map_location=torch.device('cuda:0')),
                                     strict=False)
    
    return model
    
model = _load_model()

def _preprocess(img):

    img = transform(img).unsqueeze(0)
    
    return img

def find_objects(img , text_prompt):
    
    img = _preprocess(img)
    
    with torch.no_grad():
        preds = model(img, text_prompt)[0]
        res = torch.sigmoid(preds[0][0])
    
    return res