# -*- coding: utf-8 -*-
# File: common.py

import cv2
import yaml
import base64
import numpy as np


from tensorpack.dataflow import RNGDataFlow
from tensorpack.dataflow.imgaug import ImageAugmentor, ResizeTransform


class DataFromListOfDict(RNGDataFlow):
    def __init__(self, lst, keys, shuffle=False):
        self._lst = lst
        self._keys = keys
        self._shuffle = shuffle
        self._size = len(lst)

    def __len__(self):
        return self._size

    def __iter__(self):
        if self._shuffle:
            self.rng.shuffle(self._lst)
        for dic in self._lst:
            dp = [dic[k] for k in self._keys]
            yield dp


class CustomResize(ImageAugmentor):
    """
    Try resizing the shortest edge to a certain number
    while avoiding the longest edge to exceed max_size.
    """

    def __init__(self, short_edge_length, max_size, interp=cv2.INTER_LINEAR):
        """
        Args:
            short_edge_length ([int, int]): a [min, max] interval from which to sample the
                shortest edge length.
            max_size (int): maximum allowed longest edge length.
        """
        super(CustomResize, self).__init__()
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        size = self.rng.randint(
            self.short_edge_length[0], self.short_edge_length[1] + 1)
        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, scale * w
        else:
            newh, neww = scale * h, size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return ResizeTransform(h, w, newh, neww, self.interp)



class CustomResizeT(ImageAugmentor):
    """
    Try resizing hight and width to certain numbers.
    """

    #def __init__(self, new_h, w_min_range,w_max_range, interp=cv2.INTER_LINEAR):
    def __init__(self, new_h, w_min_range,num_max_pool_layers,interp=cv2.INTER_LINEAR):
        """
        Args:
            new_h: new hight.
            w_min_range: width minimum interval to sample from 2
            w_max_range: width  maximum interval to sample from 
            
        """
        super(CustomResizeT, self).__init__()
        if isinstance(w_min_range, int):
            w_min_range = (w_min_range, w_min_range)
        #if isinstance(w_max_range, int):
        #    w_max_range = (w_max_range, w_max_range)
        self._init(locals())

    #to be tested
    def get_transform(self, img):
        h, w = img.shape[:2]
        scale = h/self.new_h
        new_w = int(w/scale)
        if new_w < self.w_min_range[0]:
            new_w=np.random.randint(self.w_min_range[0],self.w_min_range[1])
            
            
        return ResizeTransform(h, w, self.new_h, new_w, self.interp)

    
    def preprocessing(self,image):
        if isinstance(image,str):
            image= image_base64decode(image_str=image,cvs_color=1)

        image = np.array(image)
        image = image.astype("float32")

        img = self.augment(image)

        assert self.num_max_pool_layers==2
        input_length = (img.shape[1]//2) //2
        
        if len(img.shape) ==2:
                img = np.expand_dims(img, axis=0)
                img = np.expand_dims(img, axis=3)
                
        if len(img.shape) ==3:
                img = np.expand_dims(img, axis=0)
        input_length = np.array([input_length]).reshape((1,1))
        
        return img, input_length
        
def box_to_point8(boxes):
    """
    Args:
        boxes: nx4

    Returns:
        (nx4)x2
    """
    b = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]]
    b = b.reshape((-1, 2))
    return b


def point8_to_box(points):
    """
    Args:
        points: (nx4)x2
    Returns:
        nx4 boxes (x1y1x2y2)
    """
    p = points.reshape((-1, 4, 2))
    minxy = p.min(axis=1)   # nx2
    maxxy = p.max(axis=1)   # nx2
    return np.concatenate((minxy, maxxy), axis=1)

def clip_boxes(boxes, shape):
    """
    Args:
        boxes: (...)x4, float
        shape: h, w
    """
    orig_shape = boxes.shape
    boxes = boxes.reshape([-1, 4])
    h, w = shape
    boxes[:, [0, 1]] = np.maximum(boxes[:, [0, 1]], 0)
    boxes[:, 2] = np.minimum(boxes[:, 2], w)
    boxes[:, 3] = np.minimum(boxes[:, 3], h)
    return boxes.reshape(orig_shape)


def to_voc(pred_label,ivoc):

    assert isinstance(ivoc,dict), "expected dic"
    return ''.join([ ivoc[i] for i in pred_label if i in ivoc.keys()]).strip()



def load_yaml_unsafe(file):
    with open(file,'r') as file:
        config_yaml = yaml.unsafe_load(file)
    return config_yaml


def image_base64decode(image_str,cvs_color=1):

    img = base64.b64decode(image_str.encode('utf-8'))
    img = np.fromstring(img, np.uint8)
    return cv2.imdecode(img,cvs_color)

def image_base64encode(image):
    _, image = cv2.imencode('.jpeg', image)
    return base64.b64encode(image).decode('utf-8')


def label_preprocessing(transcription: str,voc: dict) -> list:
    label =[]
    for c in transcription.strip().lower():
        if c in voc.keys():
            label.append(voc[c])    
        else:
            raise Exception(" char: '{}',,, {} not all chars in vocab {}".format(c,transcription,voc))

    assert voc[''] not in label , "label contains a blank char {}".format(label)
    if len(label)== 0:
        label.append(voc[' ']) 
    assert len(label) > 0 ,"label has zero length {}".format(label)

    return label 

def crop_tag(image: np.ndarray,
                         box: np.ndarray, 
                         hight_rang_rand: list, 
                         weight_rang_rand: list,
                         width_limit: list) -> np.ndarray:
                
    box = np.array(box)
    height , width =image.shape[:2]
    if len(width_limit) != None:
        if width_limit==[-1,-1]:
            assert weight_rang_rand[0]==0
            # here the width of the tag starts from 0 to the end of the image.
            box[0]=0
            box[2]=width
            
        
    im_tag = image[int(box[1]+np.random.randint(hight_rang_rand[0],hight_rang_rand[1])):
                    int(box[3])+np.random.randint(hight_rang_rand[0],hight_rang_rand[1]), \
                    int(box[0])+np.random.randint(weight_rang_rand[0],weight_rang_rand[1]): \
                    int(box[2])+np.random.randint(weight_rang_rand[0],weight_rang_rand[1]), \
                    :]
    return im_tag
