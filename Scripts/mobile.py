# server.py
import json
from fastapi.responses import JSONResponse
from openvino.inference_engine import IECore
import cv2
import numpy as np
from fastapi import FastAPI



app = FastAPI()


ie = IECore()

def Face_Detection(frame):
    model_xml = r'models\face-detection-adas-0001\face-detection-adas-0001.xml'
    model_bin = r'models\face-detection-adas-0001\face-detection-adas-0001.bin'
    net = ie.read_network(model=model_xml, weights=model_bin)
    exec_net = ie.load_network(network=net, device_name='CPU')
    input_name = next(iter(net.input_info))
    input_info = net.input_info[input_name]
    n, c, h, w = input_info.tensor_desc.dims
    resized_frame = cv2.resize(frame, (w, h))
    input_data = resized_frame.transpose((2, 0, 1))

        # Perform inference on the frame
    output = exec_net.infer(inputs={input_name: input_data})

        # Process the output data
    output_name = next(iter(net.outputs))
    output_data = output[output_name]
    boxes = output_data[0][0]  # Assuming a single image was processed

        # Loop through the detected faces and draw bounding boxes on the frame
    Number_Of_Faces = 0
    for box in boxes:
        confidence = box[2]
        if confidence > 0.5:  # Filter detections based on confidence threshold
                Number_Of_Faces += 1
                x_min = int(box[3] * frame.shape[1])
                y_min = int(box[4] * frame.shape[0])
                x_max = int(box[5] * frame.shape[1])
                y_max = int(box[6] * frame.shape[0])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    window_width = 800  # Set the desired window width
    aspect_ratio = frame.shape[1] / frame.shape[0]
    window_height = int(window_width / aspect_ratio)
    frame = cv2.resize(frame, (window_width, window_height))    
        

    return boxes , Number_Of_Faces , frame


@app.post('/detect_faces')
async def detect_faces(request):
    frame_bytes = await request.body()
    frame_arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)

    boxes, num_faces , frame= Face_Detection(frame)

    # Prepare the response
    response_data = {
        'boxes': boxes.tolist(),
        'num_faces': num_faces,
    }

    return JSONResponse(content=response_data)


