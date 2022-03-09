try:
    from .common import CustomResizeT
    from .config import config
except ImportError:
    from common import CustomResizeT
    from config import config

import cv2
import yaml
import base64
import logging
import numpy as np
import tensorflow as tf
from google.protobuf.struct_pb2 import ListValue

from seldon_core.proto import prediction_pb2
from seldon_core.utils import get_data_from_proto, array_to_list_value , grpc_datadef_to_array


logger = logging.getLogger('_Serving_')

class Serving(object):
    
    def __init__(self,input_tensor_names,output_tensor_names,graph_path='./compact.pb',config_path='./config.yaml',prefix_name='import'):
        """
        These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest
        """
        self.prefix_name= prefix_name
        self.input_tensor_names = [name.strip() for name in input_tensor_names.split(',')]
        self.output_tensor_names = [name.strip() for name in output_tensor_names.split(',')]
        self.serving_graph= ServingGraph(graph_path,
                                        self.input_tensor_names,
                                        self.output_tensor_names,
                                        self.prefix_name)
        self.serving_graph.create_session()
        self.serving_graph.apply_compact()
        self.output_tensors = get_tensors(self.serving_graph.sess.graph,self.serving_graph.output_tensor_names,prefix_name=self.prefix_name)
        self.input_tensors = get_tensors(self.serving_graph.sess.graph,self.serving_graph.input_tensor_names,prefix_name=self.prefix_name)


        with open(config_path,'r') as f:
            self.cfg = yaml.unsafe_load(f)

        self.cfg.MODEL.INFER_SOURCES = [tower for tower in self.cfg.MODEL.INFER_SOURCES if 'ema' in tower]
        self.custom_resize = CustomResizeT(self.cfg.PREPROC.IMG.FIXED_H,
                                  self.cfg.PREPROC.W_MIN_RANGE,
                                  num_max_pool_layers=2,
                                  interp=cv2.INTER_LINEAR)
    def predict(self,X,feature_names):
        """
        inputs:
            
            X: [str,nd.array,(h, w, c),[y1, y2]]
            feature_names: ['image_id','image','image_shape','bbox']

        outputs:
            results: {'image_id': str, 
                      'image': nd.array
                      'image_shape': (h, w, c), 
                      'bbox': [y1, y2],
                       'transcription':str}
        """    
    
        res = {}
        image_id = X['image_id']
        image = X['image']
        if isinstance(image,str):
            image = base64.b64decode(image.encode('utf-8'))
        image = np.array(X['image'])
        res['image_id']=image_id
        res['image_shape']=image.shape
        res['bbox']=X['bbox']

        image = image.astype("float32")

        img = self.custom_resize.augment(image)
        #assert self.num_max_pooling == 2
        input_length = (img.shape[1]//2) //2
        
        if len(img.shape) ==2:
                img = np.expand_dims(img, axis=0)
                img = np.expand_dims(img, axis=3)
                
        if len(img.shape) ==3:
                img = np.expand_dims(img, axis=0)
        input_length = np.array([input_length]).reshape((1,1))
        
        

        logger.info("images size = {}".format(image.shape))

        output = self.serving_graph.sess.run(self.output_tensors,
                                {self.input_tensors[0]:img,
                                 self.input_tensors[1]:input_length})
        logger.info("output {}".format(output))

        
        def slice_compine(l):
            return [[l[i +1*i ],l[i +1*i  + 1]] for i in  range(len(l)//2)]
        output = slice_compine(output)
        logger.info("output {}".format(output))


        for i, (pred_label, prob)  in enumerate(output):
            res[self.cfg.MODEL.INFER_SOURCES[i]]={'transcription':to_voc(pred_label[0],self.cfg.DATA.IVOC)}

        logger.info("res {}".format(res))

        return res



class ServingGraph(object):
    
    def __init__(self,graph_path,input_tensor_names,output_tensor_names,prefix_name):
        assert isinstance(input_tensor_names,list)
        assert isinstance(output_tensor_names,list)
        self._init(locals())
        
    def create_session(self):
        self.sess =  tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True)) 
    
    def apply_compact(self):
        """Run the pruned and frozen inference graph. """
        if not self.sess:
            self.create_session()
        
        # Note, we just load the graph and do *not* need to initialize anything.
        with tf.io.gfile.GFile(self.graph_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def,name=self.prefix_name)
    
    def _init(self, params=None):
        if params:
            for k, v in params.items():
                if k != 'self' and not k.startswith('_'):
                    setattr(self, k, v)

def get_tensors(graph,names,prefix_name='import'):
    if isinstance(names,str):
        names = [names]

    tensors = []
    try:
        for name in names:
            name =  prefix_name+"/"+name  if prefix_name not in name else name
            tensors.append(graph.get_tensor_by_name(name + ':0')) 
    except KeyError:
            raise KeyError("Your model does not define the tensor '{}' in inference context.".format(name))

    return tensors


def image_base64decode(image_str):

    img = base64.b64decode(image_str.encode('utf-8'))
    img = np.fromstring(img, np.uint8)
    return cv2.imdecode(img, 1) 

def image_base64encode(image):
    _, image = cv2.imencode('.jpeg', image)
    return base64.b64encode(image).decode('utf-8')

def slice_compine(l):
        return [[l[i +1*i ],l[i +1*i  + 1]] for i in  range(len(l)//2)]


def to_voc(pred_label,ivoc):

    assert isinstance(ivoc,dict), "expected dic"
    return ''.join([ ivoc[i] for i in pred_label if i in ivoc.keys()]).strip()