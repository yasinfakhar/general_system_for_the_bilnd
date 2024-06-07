# client.py
import threading
import grpc
import camera_stream_pb2
import camera_stream_pb2_grpc
import cv2
import concurrent.futures
import numpy as np
import time
import shared_variable
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pynput import keyboard
import speech_recognition as sr
import whisper
from config import Config

model = whisper.load_model("tiny.en")

# Global variables to manage the recording statemodel = whisper.load_model("tiny")
is_recording = False
recording = []
fs = 44100  # Sample rate
stream = None  # Stream object to manage the input stream

def on_press(key):
    global is_recording, recording, stream

    try:
        if key == keyboard.Key.esc:
            if not is_recording:

                # Start recording
                print("Recording started...")
                is_recording = True
                recording = []
                sd.default.samplerate = fs
                sd.default.channels = 1
                stream = sd.InputStream(callback=callback)
                stream.start()
            else:
                # Stop recording
                print("Recording stopped...")
                is_recording = False
                stream.stop()
                stream.close()
                # Convert list to numpy array
                recording_np = np.concatenate(recording, axis=0)

                # Save the recording to a file
                wav_file = 'output.wav'
                wav.write(wav_file, fs, recording_np)

                # Clear the recording list
                recording = []
                transcript = transcribe_audio(wav_file).lower()

                print(transcript)

                if "face recognition" in transcript:
                    shared_variable.mode = "face"
                elif "segmentation" in transcript:
                    shared_variable.mode = "seg"
                elif "looking for" in transcript:
                    shared_variable.prompt = transcript.split("looking for ")[1]
                elif "exit" in transcript:
                    exit()

    except AttributeError:
        pass


def callback(indata, frames, time, status):
    global recording
    if is_recording:
        recording.append(indata.copy())


def transcribe_audio(file_path):

    result = model.transcribe("output.wav")
    return result["text"]


def asr():
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


def camera_stream(server_addr):
    channel = grpc.insecure_channel(server_addr+':50051')
    stub = camera_stream_pb2_grpc.CameraStreamStub(channel)

    # Open camera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            continue

        ret, encoded_frame = cv2.imencode(".jpg", frame)
        frame_bytes = encoded_frame.tobytes()
        frame_request = camera_stream_pb2.Message(
            image=frame_bytes, mode=shared_variable.mode, prompt=shared_variable.prompt)
        response = stub.StreamFrames(iter([frame_request]))

        for received_frame in response:
            if shared_variable.mode == 'seg':
                received_frame_np = cv2.imdecode(np.frombuffer(
                    received_frame.image, np.uint8), cv2.IMREAD_GRAYSCALE)
            else:
                received_frame_np = cv2.imdecode(np.frombuffer(
                    received_frame.image, np.uint8), cv2.IMREAD_COLOR)

            cv2.imshow('result', received_frame_np)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    asr_thread = threading.Thread(target=asr)
    asr_thread.start()
    camera_stream(server_addr=Config.server_addr)
