import PIL
import torch
from torch import nn

from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k

from .siamese import SiameseDetector
from .siamese import image_encoder, object_encoder
    
class SiameseTracker(Tracker):
    def __init__(self, weights=None, size=32):
        super(SiameseTracker, self).__init__(
            name='SiameseTracker', # name of the tracker
            is_deterministic=True   # deterministic (True) or stochastic (False)
        )
        self.sim_network = SiameseDetector()
        if weights:
            self.sim_network.load_state_dict(torch.load(weights))
        # self.im_post_processor = im_post_processing()
        
    
    def init(self, image, box):
        """Initialize your tracking model in the first frame
        
        Arguments:
            image {PIL.Image} -- Image in the first frame.
            box {np.ndarray} -- Target bounding box (4x1,
                [left, top, width, height]) in the first frame.
        """
        left, top, width, height = box
        crop = [left, top, left + width, top + height]
        self.box = box
        self.cmp_im = image.crop(crop)

    def update(self, image):
        """Locate target in an new frame and return the estimated bounding box.
        
        Arguments:
            image {PIL.Image} -- Image in a new frame.
        
        Returns:
            np.ndarray -- Estimated target bounding box (4x1,
                [left, top, width, height]) in ``image``.
        """
        sub_images = self.box_proposal(image)
        output = self.sim_network(self.cmp_im, sub_images)
        # self.box = self.im_post_processor(output)
        return self.box


def run_tracker(tracker, frames, init_box):
    tracker.init(frames[0], init_box)
    boxes = [init_box]
    for frame in frames[1:]: 
        box = tracker.update(frame)
        boxes.append(box)
    return boxes
    


