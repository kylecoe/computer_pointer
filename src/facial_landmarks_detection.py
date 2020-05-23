'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import cv2
from openvino.inference_engine import IENetwork, IECore, IEPlugin
import time

class FacialLandmarksDetection:
    '''
    Class for the Facial Landmark Detection Model.
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
        print('Landmark Load time = ', (t1-t0))

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        #print(self.input_name)
        self.check_model()


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        #t0 = time.time()
        self.frame_height = image.shape[0]
        self.frame_width = image.shape[1]

        frame = self.preprocess_input(image)

        # Change to self
        input_dict = {self.input_name: frame}


        if self.async_mode == True:
            self.net.requests[0].async_mode_infer(input_dict)
        else:
            self.net.requests[0].infer(input_dict)
        
        if self.net.requests[0].wait(-1) == 0:
            result = self.net.requests[0].outputs

        # Returns a list of the xy landmark coordinates
        eyes = self.preprocess_output(result[self.output_name])

        '''
        # Test displaying the image
        cv2.rectangle(image, (eyes[0][0], eyes[0][1]), (eyes[0][2], eyes[0][3]), (0,0,205))
        cv2.rectangle(image, (eyes[1][0], eyes[1][1]), (eyes[1][2], eyes[1][3]), (0,0,205))
        cv2.imshow('test', image)
        cv2.waitKey(0)
        '''
        eye_coords = eyes

        left_eye = image[eyes[0][1]: eyes[0][3], eyes[0][0]: eyes[0][2]]
        right_eye = image[eyes[1][1]: eyes[1][3], eyes[1][0]: eyes[1][2]]
        eyes = [left_eye, right_eye]

        #cv2.imshow('left', left_eye)
        #cv2.imshow('right', right_eye)
        #print(left_eye.shape)
        #cv2.waitKey(0)

        #t1 = time.time()
        #tt = t1 - t0
        #print('Face Landmark Inference Time = ', (t1-t0))
        return eyes, eye_coords


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''

        (n,c,h,w) = self.model.inputs[self.input_name].shape
        p_frame = cv2.resize(image, (w,h))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape((n, c, h, w))

        return p_frame

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        for item in outputs:
            x1,y1 = [item[0] * self.frame_width, item[1] * self.frame_height]   # Left eye
            x2,y2 = [item[2] * self.frame_width, item[3] * self.frame_height]   # Right eye
            x3,y3 = [item[4] * self.frame_width, item[5] * self.frame_height]
            x4,y4 = [item[6] * self.frame_width, item[7] * self.frame_height]
            x5,y5 = [item[8] * self.frame_width, item[9] * self.frame_height]

            eye1_xmin = int(x1 - 20)
            eye1_ymin = int(y1 - 20)
            eye1_xmax = int(x1 + 20)
            eye1_ymax = int(y1 + 20)

            eye2_xmin = int(x2 - 20)
            eye2_ymin = int(y2 - 20)
            eye2_xmax = int(x2 + 20)
            eye2_ymax = int(y2 + 20)

            eye1_vertices = [eye1_xmin,eye1_ymin,eye1_xmax,eye1_ymax]
            eye2_vertices = [eye2_xmin,eye2_ymin,eye2_xmax,eye2_ymax]
            eye_boxes = [eye1_vertices, eye2_vertices]

        return eye_boxes

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
