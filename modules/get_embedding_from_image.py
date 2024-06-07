import cv2
from modules.face_recognition import save_embedding
import os
from config import Config


def save_embedding_for_new_faces():

    images = os.listdir("faces")

    for image in images:

        name = image.split(".")[0]

        if (not os.path.exists(f'embeddings/{name}.npy') or Config.force_to_create_embedding == True):

            face = cv2.imread(f"faces/{image}")
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            save_embedding(face, name)

            print(f'[SUCCESS] New embedding for "{name}" saved')
