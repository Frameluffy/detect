import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torchvision
from torchvision import transforms, datasets, models
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches
import os
import streamlit as st
import tensorflow as tf
# from google.colab import drive
# drive.mount('/content/gdrive')
from streamlit_webrtc import webrtc_streamer
import av



a = st.file_uploader('upload image', type=['png', 'jpg', 'jpeg'])


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

    plt.axis("off")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    picture = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return picture

device = torch.device('cpu')
model = get_model_instance_segmentation(3)
model.load_state_dict(torch.load(r"D:\model\classifier.pt", map_location=torch.device('cpu')))


if a is not None:
    img = Image.open(a).convert("RGB")
    st.image(img)
    st.title('hello world')
    convert_tensor = transforms.ToTensor()
    a = convert_tensor(img)
    model.eval()
    with torch.no_grad():
        preds = model([a])


    demo = preds.copy()
    new_demo = dict()
    for i in demo[0].keys():
        new_demo[i] = preds[0][i][preds[0]['scores']>0.5]

    idx = 0
    print("Prediction")
    picture = plot_image_withColor([a][idx], [new_demo][idx]) # preds
    st.image(picture)

def video_frame_callback(frame):
    img = frame.to_image()
    convert_tensor = transforms.ToTensor()
    a = convert_tensor(img)
    model.eval()
    with torch.no_grad():
        preds = model([a])

    demo = preds.copy()
    new_demo = dict()
    for i in demo[0].keys():
        new_demo[i] = preds[0][i][preds[0]['scores']>0.5]

    idx = 0
    print("Prediction")
    picture = plot_image_withColor([a][idx], [new_demo][idx]) # preds
    flipped = picture[::,:,::-1]

    return av.VideoFrame.from_ndarray(flipped, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback,media_stream_constraints={"video": True, "audio": False},
    async_processing=True,)