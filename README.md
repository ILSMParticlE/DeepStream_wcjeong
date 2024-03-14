# Deepstream Video Summarization

Video summarization via face detection &amp; recognition with Deepstream

##

### Demo

#### Target image
<img src = "https://github.com/ILSMParticlE/Deepstream_wcjeong/assets/20856552/99d2d043-4260-45cc-bcf1-2f142c4d5f74" width="20%" height="20%">

##

#### Source video

https://github.com/ILSMParticlE/Deepstream_wcjeong/assets/20856552/bc4907f1-fb0f-44f2-8f0c-32fcebabd4f1

##

#### Tracked result

  
https://github.com/ILSMParticlE/Deepstream_wcjeong/assets/20856552/215e6496-b275-4da9-ae52-aae4fffd3aaa



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
