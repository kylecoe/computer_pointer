'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
import math
import numpy as np
import time
from openvino.inference_engine import IENetwork, IECore, IEPlugin

class GazeEstimation:
    '''
    Class for the Gaze Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, async_mode=False):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model = model_name
        self.device = device
        self.extensions = extensions
        self.bin = model_name + 'bin'
        self.xml = model_name + 'xml'
        self.async_mode = async_mode

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IEPlugin(device = self.device)
        self.model = IENetwork(model = self.xml, weights = self.bin)
        t0 = time.time()
        self.net = self.plugin.load(self.model)
        t1 = time.time()
        print('Gaze Estimation Load time = ', (t1-t0))

        self.input_name = [item for item in self.model.inputs.keys()]
        self.pose_shape = self.model.inputs[self.input_name[0]].shape
        self.eye_shape = self.model.inputs[self.input_name[1]].shape
        self.output_name = [item for item in self.model.outputs.keys()]
        #print('self.output_name = ', self.output_name)

        self.check_model()


    def predict(self, pose, eyes):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        
        # The left and right eye images both have the same dimension
        self.eye_height = eyes[0].shape[0]
        self.eye_width = eyes[0].shape[1]

        lefte, righte = self.preprocess_input(eyes)

        input_dict = {self.input_name[0]: pose,
                    self.input_name[1]: lefte,
                    self.input_name[2]: righte}

        if self.async_mode == True:
            self.net.requests[0].async_mode_infer(input_dict)
        else:
            self.net.requests[0].infer(input_dict)

        if self.net.requests[0].wait(-1) == 0:
            result = self.net.requests[0].outputs
        
        mouse = self.preprocess_output(result[self.output_name[0]], pose)
        
        
        #print('Gaze Inference Time = ', (t1-t0))
        return mouse

    def preprocess_input(self, image_list):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''

        (n,c,h,w) = self.model.inputs[self.input_name[1]].shape
        #print(image_list[0].shape)
        try:
            pframe_le = cv2.resize(image_list[0], (w,h))
            pframe_le = pframe_le.transpose((2,0,1))
            pframe_le = pframe_le.reshape((n, c, h, w))

            pframe_re = cv2.resize(image_list[1], (w,h))
            pframe_re = pframe_re.transpose((2,0,1))
            pframe_re = pframe_re.reshape((n, c, h, w))
        except:
            print('Head angle too extreme for model to pick up eyes / gaze')
            print('Try Again')
            exit(1)

        return pframe_le, pframe_re

    def preprocess_output(self, outputs, pose):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        roll = pose[2]
        #outputs = outputs / np.linalg.norm(outputs)
        cos = math.cos(roll * math.pi/180)
        sin = math.sin(roll * math.pi/180)

        x = outputs[0][0] * cos + outputs[0][1] * sin
        y = -outputs[0][0] * sin + outputs[0][1] * cos

        return [x, y]

    def check_model(self):
        supported_layers = self.plugin.get_supported_layers(self.model)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0:
            print('Unsupported layers found: {}'.format(unsupported_layers))
            if self.extensions != None and self.device == 'CPU':
                print('Attempting to add extension...')
                try:
                    self.plugin.add_cpu_extension(self.extensions, self.device)
                except:
                    print('Error while adding extension')
                    exit(1)

