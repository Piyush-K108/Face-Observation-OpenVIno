import kivy
kivy.require('1.0.7')
kivy.Logger.disabled=False
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import numpy as np
from threading import Thread
from openvino.inference_engine import IECore
from kivy.lang import Builder
import time

from kivy.config import Config
# Rest of your imports and code...

# Set the configuration option for hot reload
Config.set('graphics', 'allow_graphics_reloading', True)
Builder.load_string('''
<CameraBoxLayout>:
    orientation: 'vertical'
    Camera:
        id: camera
        resolution: (640, 480)
        play: True
    # Button:
    #     text: 'Detect'
    #     size_hint_y: None
    #     height: '48dp'
    #     on_press: root.face_detection()
''')


class CameraBoxLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(CameraBoxLayout, self).__init__(**kwargs)
        self.face_frame = Image()
        self.add_widget(self.face_frame)

    def capture(self):
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        camera.export_to_png("IMG_{}.png".format(timestr))
        print("Captured")

    def stop_face_detection(self):
        app = App.get_running_app()
        app.is_face_detection_running = False

    def update_texture(self, image_widget, texture):
        image_widget.texture = texture

    def get_texture(self, frame):
        # Convert the frame to RGB format and invert it
        
        
        frame_inverted = np.flipud(frame)
        # Create the texture
        buf = frame_inverted.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, bufferfmt='ubyte')
        return texture


class CameraApp(App):
    def build(self):
        self.is_face_detection_running = True
        self.camera_box_layout = CameraBoxLayout()
        self.face_detection_event = Clock.schedule_interval(self.face_detection, 1)
        return self.camera_box_layout

    def face_detection(self, dt):
        if not self.is_face_detection_running:
            self.face_detection_event.cancel()
            return

        ie = IECore()
        model_xml = r'models/face-detection-adas-0001/face-detection-adas-0001.xml'
        model_bin = r'models/face-detection-adas-0001/face-detection-adas-0001.bin'
        net = ie.read_network(model=model_xml, weights=model_bin)
        exec_net = ie.load_network(network=net, device_name='CPU')
        input_name = next(iter(net.input_info))
        input_info = net.input_info[input_name]
        n, c, h, w = input_info.tensor_desc.dims

        camera = self.camera_box_layout.ids['camera']
        texture = camera.texture
        frame = np.frombuffer(texture.pixels, dtype=np.uint8)
        frame = frame.reshape((texture.height, texture.width, 4))
        frame = frame[:, :, :3]  # Remove alpha channel

        frame2 = frame.copy()

        # Resize the frame to match the input size of the model
        resized_frame = cv2.resize(frame2, (w, h))
        input_data = resized_frame.transpose((2, 0, 1))

        # Perform inference on the frame
        output = exec_net.infer(inputs={input_name: input_data})

        # Process the output data
        output_name = next(iter(net.outputs))
        output_data = output[output_name]
        boxes = output_data[0][0]  # Assuming a single image was processed

        # Loop through the detected faces and draw bounding boxes on the frame
        for box in boxes:
            confidence = box[2]
            if confidence > 0.5:  # Filter detections based on confidence threshold
                x_min = int(box[3] * frame2.shape[1])
                y_min = int(box[4] * frame2.shape[0])
                x_max = int(box[5] * frame2.shape[1])
                y_max = int(box[6] * frame2.shape[0])
                cv2.rectangle(frame2, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Convert the frames to textures and update the image widgets
        # frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGRA2RGB)
        face_texture = self.camera_box_layout.get_texture(frame2)

        # Schedule the texture update on the main thread using Clock.schedule_once
        self.camera_box_layout.update_texture(self.camera_box_layout.face_frame, face_texture)


if __name__ == '__main__':
    CameraApp().run()
