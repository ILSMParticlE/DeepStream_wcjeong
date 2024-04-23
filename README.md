# Deepstream Video Summarization

Video summarization via face detection &amp; recognition with Deepstream

##

### Demo

#### Target image
<!--
<img src = "https://github.com/ILSMParticlE/Deepstream_wcjeong/assets/20856552/99d2d043-4260-45cc-bcf1-2f142c4d5f74" width="20%" height="20%">
-->
<img src = "https://github.com/ILSMParticlE/DeepStream_wcjeong/assets/20856552/044e73d3-4eab-4a85-adb4-66bf5b077ee5" width="20%" height="20%">

 * source : https://ko.wikipedia.org/wiki/%EC%BD%94%EB%82%9C_%EC%98%A4%EB%B8%8C%EB%9D%BC%EC%9D%B4%EC%96%B8

##

#### Source video
<!--
https://github.com/ILSMParticlE/Deepstream_wcjeong/assets/20856552/bc4907f1-fb0f-44f2-8f0c-32fcebabd4f1
  * source : https://www.pexels.com/video/people-going-in-and-out-of-the-royal-opera-house-1721303/
-->
[![Video Label](http://img.youtube.com/vi/-7tX559lsgc/0.jpg)](https://youtu.be/-7tX559lsgc)
##

#### Tracked result
 
<!--
https://github.com/ILSMParticlE/Deepstream_wcjeong/assets/20856552/f04c7312-8f54-4be4-94df-3c5098bdac31
-->


https://github.com/ILSMParticlE/DeepStream_wcjeong/assets/20856552/de8701fd-1924-4070-9ac7-99f00bfe9606



  * You can extract part of the video that contains target.
  * Lower resolution than real output due to file size limitation 



##

### Requirement

* Deepstream 6.4
  
  Reference: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html
* Deepstream Python Bindings
  
  Reference : https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
* Detection Model(YoloV8-face)
  
  Please refer to https://github.com/marcoslucianops/DeepStream-Yolo-Face/blob/master/docs/YOLOv8_Face.md
* Recognition Model(Arcface)
  
  You can get pretrained model from [Insightface](https://github.com/deepinsight/insightface), or can download pretrained ONNX file via omz_downloader

##

### Basic App Usage

#### 1. Download ONNX files and change configuration files
* For YoloV8-face, change config_yoloV8_face.txt
  ```
  [property]
  ...
  onnx-file=(YoloV8-face ONNX route)
  model-engine-file=(YoloV8-face ONNX filename)_b1_gpu0_fp32.engine
  ...
  ```

* For Arcface, change config_arcface.txt
  ```
  [property]
  ...
  onnx-file=(Arcface ONNX route)
  model-engine-file=(Arcface ONNX filename)_b1_gpu0_fp32.engine
  ...
  ```

#### 2. Make directory for target images
```
mkdir target
mv (your own images) ./target
```

#### 3. Run
**Note**: You must use absolute route of source video including 'file://'

```
python deepstream_wcjeong.py –s file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
```
