try:
    from .common import MICRCustomResize, preprocessing, image_base64decode
    from .config import config
except ImportError:
    from common import MICRCustomResize, preprocessing, image_base64decode
    from config import config
    
import cv2
import yaml
import base64
import logging
import numpy as np
import tensorflow as tf

from seldon_core.proto import prediction_pb2
from google.protobuf.struct_pb2 import ListValue
from seldon_core.utils import get_data_from_proto


logger = logging.getLogger('Serving')

class Serving(object):
    
    def __init__(self,input_tensor_names,output_tensor_names,graph_path='./compact.pb',config_path='./config.yaml',prefix_name='import'):
        """
        These will be passed at runtime from the graph definition parameters defined in your seldondeployment kubernetes resource manifest
        """
        self.prefix_name= prefix_name
        self.serving_graph= ServingGraph(graph_path,
                          [name.strip() for name in input_tensor_names.split(',')],
                          [name.strip() for name in output_tensor_names.split(',')],
                          self.prefix_name)
        self.serving_graph.create_session()
        self.serving_graph.apply_compact()
        self.output_tensors = get_tensors(self.serving_graph.sess.graph,self.serving_graph.output_tensor_names,prefix_name=self.prefix_name)
        self.input_tensors = get_tensors(self.serving_graph.sess.graph,self.serving_graph.input_tensor_names,prefix_name=self.prefix_name)

        with open(config_path,'r') as f:
            self.cfg = yaml.unsafe_load(f)
       
        self.resizer= MICRCustomResize(short_edge_length=self.cfg.HYPER.PREPROC.TRAIN_SHORT_EDGE_SIZE,
                                       interp=cv2.INTER_LINEAR)
    def predict(self,X,feature_names):
        """
        inputs:
            
            X: [str,str base64]
            feature_names: ['image_id','image']

        outputs:
            results: {'image_id': str, 
                      'image': nd.array
                      'image_shape': (h, w, c), 
                      'bbox': [y1, y2]}
        """

        logger.info("feature_names {}".format(feature_names))
        
        res = {}
        res['image_id'] =  X[0]
        image = X[1]
        #image = np.array(image)
        
        if isinstance(image,str):
            image= image_base64decode(image_str=image,cvs_color=cv2.COLOR_BGR2GRAY)


        image = image.astype("float32")
        height, width = image.shape[:2]

        crop, img_resized, scale = preprocessing(image,self.cfg.HYPER.PREPROC.CROP_RANGE, self.resizer)
                    
        box = self.serving_graph.sess.run(self.output_tensors,
                                         {self.input_tensors[0]: np.expand_dims(img_resized,axis=0)})
        box = [i+ crop for i in box[0].tolist()[0]]
        
        micr_image = image[int(box[0]+np.random.randint(-3,3)) :int(box[1]+np.random.randint(-3,3)),\
                                            int(np.random.randint(5)):int(width-np.random.randint(5)),:1]

        micr_image = np.ascontiguousarray(micr_image)
        res['image']= micr_image.tolist()
        res['image_shape']=micr_image.shape
        res['bbox'] = box

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
