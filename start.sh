source /opt/intel/openvino/bin/setupvars.sh
python3 src/main.py -f /home/kyle/Desktop/starter/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001. -ldm /home/kyle/Desktop/starter/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009. -p /home/kyle/Desktop/starter/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001. -g /home/kyle/Desktop/starter/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002. -i 'bin/demo.mp4'

