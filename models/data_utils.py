import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
import cv2

# Need to have inputs rescaled to size which is divisible by 32x32



class im_post_processing: 
    def __init__(self, iou_threhold=.5) -> None:
        pass
    
    def __call__(self, boxes):
        pass
    
if __name__ == '__main__':  
    # creating a object
    im = Image.open("../../images/image.jpg")
    bp = box_proposal()
    bp(im)