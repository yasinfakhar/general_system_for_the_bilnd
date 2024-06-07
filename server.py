# server.py

import grpc
import camera_stream_pb2
import camera_stream_pb2_grpc
import cv2
import concurrent.futures
import numpy as np
from server_utils import inference

class CameraStreamServicer(camera_stream_pb2_grpc.CameraStreamServicer):
    def StreamFrames(self, request_iterator, context):
        for frame_request in request_iterator:

            frame_bytes = frame_request.image
            prompt = frame_request.prompt
            mode = frame_request.mode

            frame_np = cv2.imdecode(np.frombuffer(
                frame_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            try:
                result = inference(frame_np, mode, prompt=prompt)
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            ret, encoded_frame = cv2.imencode(".jpg", result)

            yield camera_stream_pb2.Message(image=encoded_frame.tobytes())


def serve():
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=2))
    camera_stream_pb2_grpc.add_CameraStreamServicer_to_server(
        CameraStreamServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    serve()
