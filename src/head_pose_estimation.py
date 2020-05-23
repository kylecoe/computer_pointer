'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
from openvino.inference_engine import IENetwork, IECore, IEPlugin
import time

class HeadPose:
    '''
    Class for the Head Pose Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None, async_mode = False):
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
        print('Head Pose Load time = ', (t1-t0))

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = [item for item in self.model.outputs.keys()]

        self.check_model()


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        
        self.frame_height = image.shape[0]
        self.frame_width = image.shape[1]

        frame = self.preprocess_input(image)

        input_dict = {self.input_name: frame}

        if self.async_mode == True:
            self.net.requests[0].async_mode_infer(input_dict)
        else:
            self.net.requests[0].infer(input_dict)
        
        if self.net.requests[0].wait(-1) == 0:
            result = self.net.requests[0].outputs

        pose = self.preprocess_output(result)
        
        
        #print('Pose Inference Time = ', (t1-t0))
        return pose


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        #print(image.shape)
        (n,c,h,w) = self.model.inputs[self.input_name].shape
        p_frame = cv2.resize(image, (w,h))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape((n, c, h, w))
        #print(p_frame.shape)

        return p_frame
    
    

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        
        yaw = outputs['angle_y_fc'][0][0]
        pitch = outputs['angle_p_fc'][0][0]
        roll = outputs['angle_r_fc'][0][0]
        #print(yaw, pitch, roll)

        return [yaw, pitch, roll]

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
