class Config:
    #app setting
    skip_rate = 2
    
    #camera setting
    camera_source = 0 #"rtsp://hoopad:admin123@192.168.10.56:554/cam/realmonitor?channel=7&subtype=0"
    
    #models setting
    depth_model = "intel-isl/MiDaS"
    depth_model_version = "MiDaS_small"
    face_detection_model_path = "buffalo_s" # buffalo_s | bufallo_l
    face_regnition_model_path = 'w600k_r50.onnx' # w600k_r50.onnx | webface_r50.onnx
    
    #face recognition
    face_detection_threshold = 0.5
    cos_distance_threshold = 0.5
    force_to_create_embedding = True
    
    # facial expression
    facial_expression_model_path = "models/EfficientFace_Trained_on_RAFDB.pth.tar"
    
    #finding objects model
    finding_objects_model_path = 'models/rd64-uni-refined.pth'
    
    #age estimation
    baby_age = 2
    child_age = 12
    teenager_age = 19
    young_age = 25
    middle_age = 40
    adult_age = 60
    old_age = 10e100
    
    play_sound_after_n_sec = 10
    
    server_addr = "localhost" #"172.17.0.2"
    
