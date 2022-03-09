# -*- coding: utf-8 -*-
# File: common.py
import cv2
import base64
import numpy as np


from tensorpack.dataflow import RNGDataFlow
from tensorpack.dataflow.imgaug import ImageAugmentor, ResizeTransform
from tensorpack.utils import logger


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


class MICRCustomResize(ImageAugmentor):
    """
    Try resizing the shortest edge to a certain number
    while avoiding the longest edge to exceed max_size.
    """

    def __init__(self, short_edge_length, interp=cv2.INTER_LINEAR):
        """
        Args:
            short_edge_length ([int, int]): a [min, max] interval from which to sample the
                shortest edge length.
        """
        super(MICRCustomResize, self).__init__()
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        self._init(locals())

    def get_transform(self, img):
        h, w = img.shape[:2]
        return ResizeTransform(h, w, self.short_edge_length[0], self.short_edge_length[0], self.interp)

def preprocessing(image,height_crop_ratio,aug):

    if isinstance(image,str):
        image= image_base64decode(image_str=image,cvs_color=1)


    image = image.astype("float32")
    height, width = image.shape[:2]

    if isinstance(height_crop_ratio, int):
            height_crop_ratio = (height_crop_ratio, height_crop_ratio)
    
    crop_ratio = np.random.uniform(low=height_crop_ratio[0],high=height_crop_ratio[1])
    crop= int(height * crop_ratio)
    r = width - crop
    image = image[crop:,int(r/2):int(int(r/2) + crop),:]
    nheight, nwidth = image.shape[:2]

    tfms = aug.get_transform(image)
    resized_img = tfms.apply_image(image)
    scale = resized_img.shape[0]/image.shape[0]

    return crop, resized_img, scale

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


def polygons_to_mask(polys, height, width):
    """
    Convert polygons to binary masks.

    Args:
        polys: a list of nx2 float array. Each array contains many (x, y) coordinates.

    Returns:
        a binary matrix of (height, width)
    """
    polys = [p.flatten().tolist() for p in polys]
    assert len(polys) > 0, "Polygons are empty!"

    import pycocotools.mask as cocomask
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


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


def filter_boxes_inside_shape(boxes, shape):
    """
    Args:
        boxes: (nx4), float
        shape: (h, w)

    Returns:
        indices: (k, )
        selection: (kx4)
    """
    assert boxes.ndim == 2, boxes.shape
    assert len(shape) == 2, shape
    h, w = shape
    indices = np.where(
        (boxes[:, 0] >= 0) &
        (boxes[:, 1] >= 0) &
        (boxes[:, 2] <= w) &
        (boxes[:, 3] <= h))[0]
    return indices, boxes[indices, :]

def image_base64decode(image_str,cvs_color=1):

    img = base64.b64decode(image_str.encode('utf-8'))
    img = np.fromstring(img, np.uint8)
    return cv2.imdecode(img,cvs_color) 

def image_base64encode(image):
    _, image = cv2.imencode('.jpeg', image)
    return base64.b64encode(image).decode('utf-8')
