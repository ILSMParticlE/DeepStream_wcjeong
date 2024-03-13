# Deepstream Video Summarization

Video summarization via face detection &amp; recognition with Deepstream

##

### Demo

video

##

### Requirement

* Deepstream 6.4
* Deepstream Python Bindings
* Detection Model(YoloV8-face)
* Recognition Model(Arcface)

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
