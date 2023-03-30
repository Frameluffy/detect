import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torchvision
from torchvision import transforms, datasets, models
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import streamlit.components.v1 as components
import matplotlib.patches as patches
import os
import time
import streamlit as st
from io import BytesIO
import tensorflow as tf
import tempfile
import cv2
from threading import Thread
# from google.colab import drive
# drive.mount('/content/gdrive')
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(layout="wide")
flipped_axe = 1
times = 1

col1, col2 = st.columns(2,gap='large')

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
        Thread(target=self.update, args=()).start()
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

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    # 加载经过预训练的模型
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    # 获取分类器的输入参数的数量in_features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print("in_features:", in_features)
    # replace the pre-trained head with a new one
    # 用新的头部替换预先训练好的头部
    # 本实验的num_classes为3 
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def plot_image_withColor(img_tensor, annotation):
    
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().numpy()

    # Display the image
    ax.imshow(np.transpose(img,(1,2,0)))
    
    for (box, label) in zip(annotation["boxes"],annotation["labels"]):
        img = img_tensor.cpu().data.numpy()

        # Display the image
        # ax.imshow(np.transpose(img,(1,2,0)))
        xmin, ymin, xmax, ymax = box.cpu()
        if(label == 1):
        # Create a Rectangle patch with different colors
        #red: with mask  green: mask_weared_incorrect  blue: without mask
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        elif(label == 2):
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
        else:
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='b',facecolor='none')
        ax.add_patch(rect)
        
    plt.tight_layout(pad=0)
    plt.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    picture = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return picture

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = get_model_instance_segmentation(3)
model.load_state_dict(torch.load(r"D:\model\classifier.pt", map_location=device))

with col1:
    
    st.header("Image")
    a = st.file_uploader('Upload image', type=['png', 'jpg', 'jpeg'])
    
if a is not None:
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    img = Image.open(a).convert("RGB")
    with col1:
        st.empty()
        st.image(img)
        add = st.button("increase")
        decrease = st.button("decrease")
        #if add:      

        convert_tensor = transforms.ToTensor()
        a = convert_tensor(img).cuda()
        model.cuda()
        model.eval()
        with torch.no_grad():
            preds = model([a])

        demo = preds.copy()
        new_demo = dict()
        for i in demo[0].keys():
            new_demo[i] = preds[0][i][preds[0]['scores']>0.8]

        idx = 0
        print("Prediction")
        st.title('Mask detection')
        picture = plot_image_withColor([a][idx], [new_demo][idx]) # preds
        flipped = picture
        # with st.spinner("Loading..."):
        #     time.sleep(5)
        st.balloons()
        f2 = flipped
        # for i in range(0,img_times):
        #     print("round = ",i)
        #     f2 = np.flip(flipped,img_flipped_axe)
        #     flipped = f2
        st.image(f2)
        
        img = av.VideoFrame.from_ndarray(flipped, format="rgb24")

        omg = img.to_image()
        buf = BytesIO()
        omg.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        # scol1, scol2, scol3 = st.columns(3,gap='small')
        # with scol1:
        #     vimgflip = st.button("Flip image vertical")
        #     if vimgflip:
        #         print("v click")
        #         if img_flipped_axe == 1:
        #             img_flipped_axe = 0
        #         elif img_flipped_axe == 0 and img_times == 2:
        #             img_times = 1
        #         elif img_flipped_axe == 0 and img_times == 1:
        #             img_times = 2
        #         st.pyplot()

        
        # with scol2:
        #     himgflip = st.button("Flip image Horizontal")
        #     if himgflip:
        #         print("h click")
        #         if img_flipped_axe == 0:
        #             img_flipped_axe = 1
        #         elif img_flipped_axe == 1 and img_times == 2:
        #             img_times = 1
        #         elif img_flipped_axe == 1 and img_times == 1:
        #             img_times = 2
        # with scol3:
        #     btn = st.download_button(
        #         label="Download Image",
        #         data=byte_im,
        #         file_name="mask-detect.png",
        #         mime="image/jpeg",
        #     )

# def video_frame_callback(frame):
#     img = frame.to_ndarray(format="rgb24")
#     convert_tensor = transforms.ToTensor()
#     a = convert_tensor(img).cuda()
#     model.cuda()
#     model.eval()
#     with torch.no_grad():
#         preds = model([a])

#     demo = preds.copy()
#     new_demo = dict()
#     for i in demo[0].keys():
#         new_demo[i] = preds[0][i][preds[0]['scores']>0.8]

#     idx = 0
#     print("Prediction")
#     picture = plot_image_withColor([a][idx], [new_demo][idx]) # preds
#     print("axe : ",flipped_axe)
#     print("times : ",times)
#     for i in range(0,times):
#         print("round = ",i)
#         flipped = np.flip(picture,flipped_axe)

#     return av.VideoFrame.from_ndarray(img, format="rgb24")

with col2:
    st.header("Realtime-camera")
    #webrtc_streamer(key="example", video_frame_callback=video_frame_callback,media_stream_constraints={"video": True, "audio": False},async_processing=True,)
    wc = st.button("WebCam")
    stframe = st.empty()
    scol1, scol2, scol3 = st.columns(3,gap='small')
       
    tffile = tempfile.NamedTemporaryFile(delete=False)
    
    if wc:
        webcam_stream = WebcamStream(stream_id=0) # 0 id for main camera
        webcam_stream.start()
        vid = cv2.VideoCapture(0)
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        with scol1:
            vflip = st.button("Flip video vertical")
        with scol2:
            hflip = st.button("Flip video Horizontal")
        with scol3:
            option = st.selectbox('choose',('play', 'stop'))

        prevTime = 0
        while option != 'stop':
            
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
                new_demo[i] = preds[0][i][preds[0]['scores']>0.8]

            idx = 0
            print("Prediction")
            frame = plot_image_withColor([a][idx], [new_demo][idx]) # preds
            stframe.image(frame,channels = 'RGB',use_column_width=True)
        print('Program Stopped')
        webcam_stream.stop()



        



