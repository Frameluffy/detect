import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import matplotlib.patches as patches
from playsound import playsound
import threading

def alert():
    threading.Thread(target=playsound, args=(r'D:\detect\sound\sound.wav',), daemon=True).start()

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
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
            text = "With mask"
            color = "#8fce00"
        elif(label == 2):
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
            text = "Without mask"
            color = "#f44336"
            print(st.session_state['playsound'])
            if st.session_state['playsound']%30 == 0:
                alert()
            st.session_state['playsound'] += 1
        else:
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='b',facecolor='none')
            text = "Mask weared incorrect"
            color = "#2986cc"
            
        ax.add_patch(rect)
        ax.text(xmin+10, ymin+20, text,fontfamily="sans-serif",color=color)
    
    plt.axis("off")   
    
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    picture = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return picture