import numpy as np
from PIL import Image, ImageOps
import cv2



class box_proposal: 
    def __init__(self, M=5, N=9) -> None:
        """
        Pipeline for proposing sub_boxes to detector
        Args:
            M (int, optional): Number of boxes in horizontal direction. Defaults to 16.
            N (int, optional): Number of boxes in vertical. Defaults to 9.
        """
        self.M = M
        self.N = N
        
    def __call__(self, image):
        """ Returns sub_boxes on a given frame

        Args:
            image (PIL.Image): Image that needs to be divided into sub_boxes
        """
        boxes = []
        width, height, channels = np.shape(image)
        print(np.shape(image))
        
        for i in range(self.M): 
            for j in range(self.N): 
                boxes.append(image.crop((i/self.M*width, j/self.N * height, (i+1)/self.M*width, (j+1)/self.N * height)))
        return boxes

class post_processing: 
    def __init__(self, iou_threhold, ) -> None:
        pass
    
    

if __name__ == '__main__':  
    # creating a object
    im = Image.open("../../images/image.jpg")
    bp = box_proposal()
    bp(im)