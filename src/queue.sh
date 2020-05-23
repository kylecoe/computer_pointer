#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

mkdir -p /output

OUTPUT_FILE=$1
DEVICE=$2
PRECISION=$3
INPUT_FILE=$4

if "[$PRECISION" == "FP32"]; then
    FACEMODELPATH=../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.
    POSEMODELPATH=../intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.
    LANDMARKSMODELPATH=../intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.
    GAZEMODELPATH=../intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.       
else
    FACEMODELPATH=../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.
    POSEMODELPATH=../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.
    LANDMARKSMODELPATH=../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.   
    GAZEMODELPATH=../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.     
fi

if echo "$DEVICE" | grep -q "FPGA"; then

    export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2

    source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
    aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2020-2_PL2_FP16_MobileNet_Clamp.aocx

    export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3


# Run the load model python script
python3 main.py  -f ${FACEMODELPATH} \
                -ldm ${LANDMARKSMODELPATH} \
                -p ${POSEMODELPATH} \
                -g ${GAZEMODELPATH} \
                -i ${INPUT_FILE} \
                -d ${DEVICE}

cd /output

tar zcvf output.tgz stdout.log stderr.log
