import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from config import Config
from insightface.model_zoo import model_zoo
import os

app = FaceAnalysis(name = Config.face_detection_model_path, allowed_modules=['detection', 'genderage'], providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0,det_thresh=Config.face_detection_threshold, det_size=(640, 640))

rec_model = model_zoo.get_model(f'models/{Config.face_regnition_model_path}')

def face_recognition(image):
    return app.get(image) , app

def get_embedding(img , face):
    return rec_model.get(img, face)

def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

def save_embedding(img , name):
    
    faces , app = face_recognition(img)
    
    if(len(faces) != 1):
        assert Exception('[ERROR] Each imgage should contain only one face')
        
    face_embadding = get_embedding(img , faces[0])
    
    np.save(f'embeddings/{name}.npy', face_embadding)
    
def load_embeddings():
    
    embeddings = os.listdir("embeddings")
    people = {}
    for embedding in embeddings:

        data = np.load(f'embeddings/{embedding}')
        name = embedding.split('.')[0]
        people[name] = data
    
    return people
        
        