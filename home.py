from torchvision import transforms
import torch
from PIL import Image
import streamlit as st
from io import BytesIO
import tempfile
import cv2
import av
from threading import Thread
import numpy as np

#from files 
from model import get_model_instance_segmentation
from plot import plot_image_withColor


class WebcamStream :
    # initialization method 
    def __init__(self, stream_id=0):
        self.stream_id = stream_id # default is 0 for main camera 
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5)) # hardware fps
        print("FPS of input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)
        # self.stopped is initialized to False 
        self.stopped = False
        # thread instantiation  
        # self.t = Thread(target=self.update, args=())
        # self.t.daemon = True # daemon threads run in background 
        
    # method to start thread 
    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        
        return self
    # method passed to thread to read next available frame  
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
    # method to return latest read frame 
    def read(self):
        return self.frame
    # method to stop reading frames 
    def stop(self):
        self.vcap.release()
        self.stopped = True

app_mode = st.sidebar.selectbox('Choose the App mode',
['Run on Image','Run on Video']
)
st.sidebar.markdown('---')
st.sidebar.title('Mask-Dection')
st.sidebar.subheader('Parameters')
detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.8)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model_instance_segmentation(3)

# url = "https://drive.google.com/file/d/1MmNGQrw1sIlBf5KL3VBQnWhInPePSeF4/view?usp=share_link"
# output = "classifier.pt"
# gdown.download(url, output, quiet=False)
# model.load_state_dict(torch.load(r"/classifier.pt", map_location=device))
model.load_state_dict(torch.load(r"D:\model\classifier.pt", map_location=device))

if app_mode =='Run on Image':
    
    st.header("Image")
    a = st.file_uploader('Upload image', type=['png', 'jpg', 'jpeg'])
    
    if a is not None:
        print(torch.cuda.is_available())
        print(torch.cuda.get_device_name(0))
        img = Image.open(a).convert("RGB")
        img = np.asarray(img)
        img_section = st.empty()
        img_section.image(img)

        col1 ,col2 ,col3 = st.columns(3,gap="small")
        with col1:
            f_horiz = st.button("Flip Horizontal")
        with col2:
            f_verti = st.button("Flip Vertical")
        with col3:
            blur = st.button("Blur")
            
        predict = st.button("Predict")

        if f_horiz:
            img = cv2.flip(img,1)
            st.session_state['img'] = img
            img_section.image(img)
        elif f_verti:
            img = cv2.flip(img,0)
            st.session_state['img'] = img
            img_section.image(img)
        elif blur:
            img = cv2.blur(img,(30,30))
            st.session_state['img'] = img
            img_section.image(img)

        if predict:
            convert_tensor = transforms.ToTensor()
            if 'img' not in st.session_state:
                print('hello')
                st.session_state['img'] = img

            img = st.session_state['img']
            a = convert_tensor(img).cuda()
            model.cuda()
            model.eval()
            with torch.no_grad():
                preds = model([a])

            demo = preds.copy()
            new_demo = dict()
            for i in demo[0].keys():
                new_demo[i] = preds[0][i][preds[0]['scores']>detection_confidence]

            idx = 0
            print("Prediction")
            st.title('Mask detection')
            picture = plot_image_withColor([a][idx], [new_demo][idx]) # preds
            # with st.spinner("Loading..."):
            #     time.sleep(5)
            st.balloons()
            st.image(picture)
            
            img = av.VideoFrame.from_ndarray(picture, format="rgb24")

            omg = img.to_image()
            buf = BytesIO()
            omg.save(buf, format="JPEG")
            byte_im = buf.getvalue()

            del st.session_state['img']
        
elif app_mode == "Run on Video":
    
    if 'object' in st.session_state:
        print('asfasfafasfafasf')
        webcam_stream = st.session_state['object']
    st.header("Realtime-camera")
    #webrtc_streamer(key="example", video_frame_callback=video_frame_callback,media_stream_constraints={"video": True, "audio": False},async_processing=True,)
    wc = st.button("WebCam")
    stframe = st.empty()
    scol1, scol2 = st.columns(2,gap='small')
    st.session_state['control']=0
    
    tffile = tempfile.NamedTemporaryFile(delete=False)
    # if st.session_state.keys == 0:
    if 'status' in st.session_state:
        print('helloooo')
        print(st.session_state['status'])
        if st.session_state['status'] == 'stop':
                print('Program Stopped')
                webcam_stream.stop()
                del webcam_stream
                
 
    if wc:
        webcam_stream = WebcamStream(stream_id=0) # 0 id for main camera
        st.session_state['object'] = webcam_stream
        st.session_state['status'] = 'play'
        webcam_stream.start()
        
        
        with scol2:
            option = st.selectbox('choose',('stop','stop!'))
            if option:
                st.session_state['status'] = 'stop'
            

        while webcam_stream.grabbed:    
            frame = webcam_stream.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = True
            convert_tensor = transforms.ToTensor()
            a = convert_tensor(frame).cuda()
            model.cuda()
            model.eval()
            with torch.no_grad():
                preds = model([a])

            demo = preds.copy()
            new_demo = dict()
            for i in demo[0].keys():
                new_demo[i] = preds[0][i][preds[0]['scores']>detection_confidence]

            idx = 0
            frame = plot_image_withColor([a][idx], [new_demo][idx]) # preds
            stframe.image(frame,channels = 'RGB',use_column_width=True)

        del st.session_state['object']
        del st.session_state['status']



        



