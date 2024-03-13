import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

import os
import sys
import time
import argparse
import platform
from ctypes import *
import numpy as np
import cv2

sys.path.append('/opt/nvidia/deepstream/deepstream/lib')
import pyds


SOURCE = ''
CONFIG_INFER_PGIE = 'config_yoloV8_face.txt'
CONFIG_INFER_SGIE = 'config_arcface.txt'
STREAMMUX_BATCH_SIZE = 1
STREAMMUX_WIDTH = 1920
STREAMMUX_HEIGHT = 1080
GPU_ID = 0
PERF_MEASUREMENT_INTERVAL_SEC = 5
SIM_THRESHOLD = 0.4
TARGET_IMG_DIR = "./target"



FACE_POOL = []
SIM_FACE_OBJ_ID = []

# for OpenCv video capture
OUTPUT_FRAME_LIST = []
OUTPUT_BBOX_LIST = []


def set_custom_bbox(obj_meta):
    border_width = 6
    font_size = 18
    x_offset = int(min(STREAMMUX_WIDTH - 1, max(0, obj_meta.rect_params.left - (border_width / 2))))
    y_offset = int(min(STREAMMUX_HEIGHT - 1, max(0, obj_meta.rect_params.top - (font_size * 2) + 1)))

    obj_meta.rect_params.border_width = border_width
    obj_meta.rect_params.border_color.red = 0.0
    obj_meta.rect_params.border_color.green = 0.0
    obj_meta.rect_params.border_color.blue = 1.0
    obj_meta.rect_params.border_color.alpha = 1.0
    obj_meta.text_params.font_params.font_name = 'Ubuntu'
    obj_meta.text_params.font_params.font_size = font_size
    obj_meta.text_params.x_offset = x_offset
    obj_meta.text_params.y_offset = y_offset
    obj_meta.text_params.font_params.font_color.red = 1.0
    obj_meta.text_params.font_params.font_color.green = 1.0
    obj_meta.text_params.font_params.font_color.blue = 1.0
    obj_meta.text_params.font_params.font_color.alpha = 1.0
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.red = 0.0
    obj_meta.text_params.text_bg_clr.green = 0.0
    obj_meta.text_params.text_bg_clr.blue = 1.0
    obj_meta.text_params.text_bg_clr.alpha = 1.0


def get_sim(f1, f2):
    t1 = f1.ravel()
    t2 = f2.ravel()
    sim = np.dot(t1, t2) / (np.linalg.norm(t1) * np.linalg.norm(t2))
    return sim


def osd_sink_pad_buffer_probe(pad, info, user_data):
    buf = info.get_buffer()
    if not buf:
        print("Unable to get buffer")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    l_frame = batch_meta.frame_meta_list
    cur_batch_frame_cnt = 0
    global FACE_POOL, SIM_FACE_OBJ_ID, OUTPUT_BBOX_LIST

    ''' For openCv video creation'''
    bboxes = []
    ''' End '''

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        remove_flag = True
        obj_cnt = 0
        user_cnt = 0

        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            obj_cnt += 1
            if obj_meta.object_id in SIM_FACE_OBJ_ID:
                remove_flag = False
                #set_custom_bbox(obj_meta)

                ''' For opencv video creation '''
                rect = obj_meta.rect_params
                bboxes.append([rect.left, rect.top, rect.left+rect.width, rect.top+rect.height])
                ''' End '''
            
            l_obj_user = obj_meta.obj_user_meta_list

            while l_obj_user:
                try:
                    obj_user_meta = pyds.NvDsUserMeta.cast(l_obj_user.data)
                except StopIteration:
                    break

                if obj_user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(obj_user_meta.user_meta_data)
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)

                    face_output = []
                    for i in range(layer.dims.d[0]):
                        face_output.append(pyds.get_detections(layer.buffer, i))
                    face_tensor = np.reshape(face_output, (512, -1))

                    for face in FACE_POOL:
                        sim = get_sim(face_tensor, face)
                        if sim > SIM_THRESHOLD:
                            remove_flag = False
                            #set_custom_bbox(obj_meta)
                            rect = obj_meta.rect_params
                            bboxes.append([rect.left, rect.top, rect.left+rect.width, rect.top+rect.height])
                            if obj_meta.object_id not in SIM_FACE_OBJ_ID:
                                SIM_FACE_OBJ_ID.append(obj_meta.object_id)
                            #display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                            #pyds.nvds_remove_display_meta_from_frame(frame_meta, display_meta)
                            #pyds.nvds_clear_display_meta_list(frame_meta)
                            #set_custom_bbox(obj_meta)
                            break
                user_cnt += 1

                try:
                    l_obj_user = l_obj_user.next
                except StopIteration:
                    break

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        #print("frame : %d, obj_cnt : %d, user_cnt ; %d" % (frame_meta.frame_num, obj_cnt, user_cnt))
        try:
            l_frame = l_frame.next
            if remove_flag:
                pyds.nvds_remove_frame_meta_from_batch(batch_meta, frame_meta)
            else:
                cur_batch_frame_cnt += 1
                global output_frame_num, OUTPUT_FRAME_LIST
                #output_frame_num += 1

                ''' For OpenCv video creation '''
                OUTPUT_FRAME_LIST.append(frame_meta.frame_num)
                OUTPUT_BBOX_LIST.append(bboxes)
                ''' End '''
                
        except StopIteration:
            break

    '''
    # search from past frame
    l_user = batch_meta.batch_user_meta_list
    
    while l_user:
        try:
            user_meta = pyds.NvDsUserMeta.cast(l_user.data)
        except StopIteration:
            break
        
        
        if user_meta and user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDS_TRACKER_PAST_FRAME_META:
            try:
                past_data_batch = pyds.NvDsTargetMiscDataBatch.cast(user_meta.user_meta_data)
            except StopIteration:
                break

            for misc_data_stream in pyds.NvDsTargetMiscDataBatch.list(past_data_batch):
                #print("streamId = ", misc_data_stream.streamID)
                #print("surfaceStreamID = ", misc_data_stream.surfaceStreamID)
                continue

                for misc_data_obj in pyds.NvDsTargetMiscDataStream.list(misc_data_stream):
                    #f.write("numobj = %d\n" % (misc_data_obj.numObj))
                    continue
                    
                    for misc_data_frame in pyds.NvDsTargetMiscDataObject.list(misc_data_obj):
                        #f.write("frameNum = %d\n" %(misc_data_frame.frameNum))
                        #f.write("confidence = %f\n" %(misc_data_frame.confidence))
                        continue


        try:
            l_user = l_user.next
        except StopIteration:
            break
    '''
    if not cur_batch_frame_cnt:
        return Gst.PadProbeReturn.DROP
    return Gst.PadProbeReturn.OK


def osd_sink_pad_buffer_probe_face_pool(pad, info, user_data):
    buf = info.get_buffer()
    if not buf:
        print("Unable to get buffer")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    l_frame = batch_meta.frame_meta_list

    while l_frame:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        l_obj = frame_meta.obj_meta_list
        obj_cnt = 0
        
        while l_obj:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            obj_cnt += 1
            l_obj_user = obj_meta.obj_user_meta_list

            while l_obj_user:
                try:
                    obj_user_meta = pyds.NvDsUserMeta.cast(l_obj_user.data)
                except StopIteration:
                    break

                if obj_user_meta.base_meta.meta_type == pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(obj_user_meta.user_meta_data)
                    layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)

                    face_output = []
                    for i in range(layer.dims.d[0]):
                        face_output.append(pyds.get_detections(layer.buffer, i))
                    face_tensor = np.reshape(face_output, (512, -1))
                    global FACE_POOL
                    FACE_POOL.append(face_tensor)

                try:
                    l_obj_user = l_obj_user.next
                except StopIteration:
                    break

            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def decodebin_child_added(child_proxy, Object, name, user_data):
    if name.find('decodebin') != -1:
        Object.connect('child-added', decodebin_child_added, user_data)
    if name.find('nvv4l2decoder') != -1:
        Object.set_property('drop-frame-interval', 0)
        Object.set_property('num-extra-surfaces', 1)
        if is_aarch64():
            Object.set_property('enable-max-performance', 1)
        else:
            Object.set_property('cudadec-memtype', 0)
            Object.set_property('gpu-id', GPU_ID)


def cb_newpad(decodebin, pad, user_data):
    streammux_sink_pad = user_data
    caps = pad.get_current_caps()
    if not caps:
        caps = pad.query_caps()
    structure = caps.get_structure(0)
    name = structure.get_name()
    features = caps.get_features(0)
    if name.find('video') != -1:
        if features.contains('memory:NVMM'):
            if pad.link(streammux_sink_pad) != Gst.PadLinkReturn.OK:
                sys.stderr.write('ERROR: Failed to link source to streammux sink pad\n')
        else:
            sys.stderr.write('ERROR: decodebin did not pick NVIDIA decoder plugin')


def create_uridecode_bin(stream_id, uri, streammux):
    bin_name = 'source-bin-%04d' % stream_id
    bin = Gst.ElementFactory.make('uridecodebin', bin_name)
    if 'rtsp://' in uri:
        pyds.configure_source_for_ntp_sync(bin)
    bin.set_property('uri', uri)
    pad_name = 'sink_%u' % stream_id
    streammux_sink_pad = streammux.get_request_pad(pad_name)
    bin.connect('pad-added', cb_newpad, streammux_sink_pad)
    bin.connect('child-added', decodebin_child_added, 0)
    return bin, streammux_sink_pad


def bus_call(bus, message, user_data):
    loop = user_data
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write('DEBUG: EOS\n')
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write('WARNING: %s: %s\n' % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write('ERROR: %s: %s\n' % (err, debug))
        loop.quit()
    return True


def is_aarch64():
    return platform.uname()[4] == 'aarch64'


def get_face_pool():
    Gst.init(None)

    loop = GLib.MainLoop()

    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write('ERROR: Failed to create pipeline\n')
        sys.exit(1)

    streammux = Gst.ElementFactory.make('nvstreammux', 'nvstreammux')
    if not streammux:
        sys.stderr.write('ERROR: Failed to create nvstreammux\n')
        sys.exit(1)
    pipeline.add(streammux)

    targets = os.listdir(TARGET_IMG_DIR)
    if not len(targets):
        sys.stderr.write('ERROR: No target images\n')
        sys.exit(1)

    pgie = Gst.ElementFactory.make('nvinfer', 'pgie')
    if not pgie:
        sys.stderr.write('ERROR: Failed to create nvinfer pgie\n')
        sys.exit(1)

    sgie = Gst.ElementFactory.make('nvinfer', 'sgie')
    if not sgie:
        sys.stderr.write('ERROR: Failed to create nvinfer sgie\n')
        sys.exit(1)

    converter = Gst.ElementFactory.make('nvvideoconvert', 'converter1')
    if not converter:
        sys.stderr.write('ERROR: Failed to create nvvideoconvert\n')
        sys.exit(1)

    osd = Gst.ElementFactory.make('nvdsosd', 'nvdsosd')
    if not osd:
        sys.stderr.write('ERROR: Failed to create nvdsosd\n')
        sys.exit(1)

    sink = None
    if is_aarch64():
        sink = Gst.ElementFactory.make('fakesink', 'nv3dsink')
        if not sink:
            sys.stderr.write('ERROR: Failed to create nv3dsink\n')
            sys.exit(1)
    else:
        sink = Gst.ElementFactory.make('fakesink', 'nveglglessink')
        if not sink:
            sys.stderr.write('ERROR: Failed to create nveglglessink\n')
            sys.exit(1)

    streammux.set_property('batch-size', STREAMMUX_BATCH_SIZE)
    streammux.set_property('batched-push-timeout', 25000)
    streammux.set_property('enable-padding', 0)
    streammux.set_property('attach-sys-ts', 1)
    pgie.set_property('config-file-path', CONFIG_INFER_PGIE)
    pgie.set_property('qos', 0)
    sgie.set_property('config-file-path', CONFIG_INFER_SGIE)
    sgie.set_property('qos', 0)
    osd.set_property('process-mode', int(pyds.MODE_GPU))
    osd.set_property('qos', 0)
    sink.set_property('async', 0)
    sink.set_property('sync', 1)
    sink.set_property('qos', 0)

    if not is_aarch64():
        streammux.set_property('nvbuf-memory-type', 0)
        streammux.set_property('gpu_id', GPU_ID)
        pgie.set_property('gpu_id', GPU_ID)
        converter.set_property('nvbuf-memory-type', 0)
        converter.set_property('gpu_id', GPU_ID)
        osd.set_property('gpu_id', GPU_ID)


    pipeline.add(pgie)
    pipeline.add(sgie)
    pipeline.add(converter)
    pipeline.add(osd)
    pipeline.add(sink)

    streammux.link(pgie)
    pgie.link(sgie)
    sgie.link(converter)
    converter.link(osd)
    osd.link(sink)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', bus_call, loop)

    osd_sink_pad = osd.get_static_pad("sink")
    if not osd_sink_pad:
        sys.stderr.write("Unable to get sink pd of osd\n")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe_face_pool, 0)
    
    ABS_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    TARGET_PATH = 'file://' + ABS_CUR_DIR + TARGET_IMG_DIR[1:] + '/'
    for idx in range(len(targets)):
        source_bin, streammux_pad = create_uridecode_bin(idx + 1, TARGET_PATH + targets[idx], streammux)
        if not source_bin:
            sys.stderr.write('ERROR: Failed to create source_bin %d\n' % idx)
            sys.exit(1)
        pipeline.add(source_bin)
        img_height, img_width, img_layer = cv2.imread(TARGET_IMG_DIR + '/' + targets[idx]).shape

        sys.stdout.write('\n#################################################################\n\n')
        sys.stdout.write('img %d, %s : %d x %d\n\n' % (idx, targets[idx], img_width, img_height))
        sys.stdout.write('#################################################################\n\n')

        streammux.set_property('width', img_width)
        streammux.set_property('height', img_height)

        pipeline.set_state(Gst.State.PLAYING)

        try:
            loop.run()
        except:
            pass

        pipeline.set_state(Gst.State.NULL)
        
        streammux.release_request_pad(streammux_pad)
        pipeline.remove(source_bin)

    
    sys.stdout.write('\n#################################################################\n\n')
    sys.stdout.write("Successfully executed on %d targets\n\n" % len(FACE_POOL))
    sys.stdout.write('#################################################################\n\n')
    sys.stdout.write('\n')


def summarize():
    Gst.init(None)

    loop = GLib.MainLoop()

    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write('ERROR: Failed to create pipeline\n')
        sys.exit(1)

    streammux = Gst.ElementFactory.make('nvstreammux', 'nvstreammux')
    if not streammux:
        sys.stderr.write('ERROR: Failed to create nvstreammux\n')
        sys.exit(1)
    pipeline.add(streammux)

    source_bin, streammux_pad = create_uridecode_bin(0, SOURCE, streammux)
    if not source_bin:
        sys.stderr.write('ERROR: Failed to create source_bin\n')
        sys.exit(1)
    pipeline.add(source_bin)

    pgie = Gst.ElementFactory.make('nvinfer', 'pgie')
    if not pgie:
        sys.stderr.write('ERROR: Failed to create nvinfer pgie\n')
        sys.exit(1)

    sgie = Gst.ElementFactory.make('nvinfer', 'sgie')
    if not sgie:
        sys.stderr.write('ERROR: Failed to create nvinfer sgie\n')
        sys.exit(1)

    tracker = Gst.ElementFactory.make('nvtracker', 'nvtracker')
    if not tracker:
        sys.stderr.write('ERROR: Failed to create nvtracker\n')
        sys.exit(1)

    converter = Gst.ElementFactory.make('nvvideoconvert', 'converter1')
    if not converter:
        sys.stderr.write('ERROR: Failed to create nvvideoconvert\n')
        sys.exit(1)

    osd = Gst.ElementFactory.make('nvdsosd', 'nvdsosd')
    if not osd:
        sys.stderr.write('ERROR: Failed to create nvdsosd\n')
        sys.exit(1)


    ''' For filesink '''
    converter2 = Gst.ElementFactory.make("nvvideoconvert", "converter2")
    
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    caps = Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)

    encoder = Gst.ElementFactory.make("avenc_mpeg4", "encoder")
    encoder.set_property("bitrate", 2000000)

    codeparser = Gst.ElementFactory.make("mpeg4videoparse", "mpeg4-parser")

    container = Gst.ElementFactory.make("qtmux", "qtmux")

    
    if not converter2:
        sys.stderr.write("ERROR: Failed to create converter2\n")
    
    if not capsfilter:
        sys.stderr.write("ERROR: Failed to create capsfilter\n")

    if not encoder:
        sys.stderr.write("ERROR: Failed to create encoder\n")

    if not codeparser:
        sys.stderr.write("ERROR: Failed to create codeparser\n")

    if not container:
        sys.stderr.write("ERROR: Failed to create container\n")
    
    
    # Create sink
    sink = None
    if is_aarch64():
        sink = Gst.ElementFactory.make('filesink', 'nv3dsink')
        sink.set_property("location", "./out.mp4")
        if not sink:
            sys.stderr.write('ERROR: Failed to create nv3dsink\n')
            sys.exit(1)
    else:
        sink = Gst.ElementFactory.make('filesink', 'nveglglessink')
        sink.set_property("location", "./out.mp4")
        if not sink:
            sys.stderr.write('ERROR: Failed to create nveglglessink\n')
            sys.exit(1)
    ''' End '''
    

    ''' For opencv video creation '''
    if 'file://' in SOURCE:
        vcap = cv2.VideoCapture(SOURCE[7:])
        global STREAMMUX_WIDTH, STREAMMUX_HEIGHT
        STREAMMUX_WIDTH = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        STREAMMUX_HEIGHT = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #vcap.release()
    else:
        sys.stderr.write("Something wrong for source\n")
        sys.exit(1)
    ''' End '''


    sys.stdout.write('\n')
    sys.stdout.write('SOURCE: %s\n' % SOURCE)
    sys.stdout.write('STREAMMUX_BATCH_SIZE: %d\n' % STREAMMUX_BATCH_SIZE)
    sys.stdout.write('STREAMMUX_WIDTH: %d\n' % STREAMMUX_WIDTH)
    sys.stdout.write('STREAMMUX_HEIGHT: %d\n' % STREAMMUX_HEIGHT)
    sys.stdout.write('GPU_ID: %d\n' % GPU_ID)
    sys.stdout.write('PERF_MEASUREMENT_INTERVAL_SEC: %d\n' % PERF_MEASUREMENT_INTERVAL_SEC)
    sys.stdout.write('JETSON: %s\n' % ('TRUE' if is_aarch64() else 'FALSE'))
    sys.stdout.write('\n')


    streammux.set_property('batch-size', STREAMMUX_BATCH_SIZE)
    streammux.set_property('batched-push-timeout', 25000)
    streammux.set_property('width', STREAMMUX_WIDTH)
    streammux.set_property('height', STREAMMUX_HEIGHT)
    streammux.set_property('enable-padding', 0)
    streammux.set_property('live-source', 1)
    streammux.set_property('attach-sys-ts', 1)
    pgie.set_property('config-file-path', CONFIG_INFER_PGIE)
    pgie.set_property('qos', 0)
    sgie.set_property('config-file-path', CONFIG_INFER_SGIE)
    sgie.set_property('qos', 0)
    tracker.set_property('tracker-width', 640)
    tracker.set_property('tracker-height', 384)
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file',
                         '/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_NvDCF_perf.yml')
    tracker.set_property('display-tracking-id', 1)
    tracker.set_property('qos', 0)
    osd.set_property('process-mode', int(pyds.MODE_GPU))
    osd.set_property('qos', 0)
    osd.set_property('display-text', 0)
    osd.set_property('display-bbox', 0)
    osd.set_property('display-mask', 0)
    sink.set_property('async', 0)
    sink.set_property('sync', 1)
    sink.set_property('qos', 0)

    if 'file://' in SOURCE:
        streammux.set_property('live-source', 0)

    if tracker.find_property('enable_batch_process') is not None:
        tracker.set_property('enable_batch_process', 1)

    if tracker.find_property('enable_past_frame') is not None:
        tracker.set_property('enable_past_frame', 1)

    if not is_aarch64():
        streammux.set_property('nvbuf-memory-type', 0)
        streammux.set_property('gpu_id', GPU_ID)
        pgie.set_property('gpu_id', GPU_ID)
        tracker.set_property('gpu_id', GPU_ID)
        converter.set_property('nvbuf-memory-type', 0)
        converter.set_property('gpu_id', GPU_ID)
        osd.set_property('gpu_id', GPU_ID)


    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie)
    pipeline.add(converter)
    pipeline.add(converter2)
    pipeline.add(encoder)
    pipeline.add(capsfilter)
    pipeline.add(codeparser)
    pipeline.add(container)
    pipeline.add(osd)
    pipeline.add(sink)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie)
    sgie.link(converter)
    converter.link(osd)
    osd.link(converter2)
    converter2.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(codeparser)

    container_video_sink_pad = container.get_request_pad("video_0")
    if not container_video_sink_pad:
        sys.stderr.write("ERROR: Unable to get sinkpad of qtmux\n")

    codeparser_src_pad = codeparser.get_static_pad("src")
    if not codeparser_src_pad:
        sys.stderr.write("ERROR : Unable go get mpeg4 parse source pad\n")
    
    codeparser_src_pad.link(container_video_sink_pad)
    container.link(sink)


    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect('message', bus_call, loop)

    osd_sink_pad = osd.get_static_pad("sink")
    if not osd_sink_pad:
        sys.stderr.write("Unable to get sink pd of osd\n")
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    pipeline.set_state(Gst.State.PLAYING)

    sys.stdout.write('\n')

    try:
        loop.run()
    except:
        pass

    pipeline.set_state(Gst.State.NULL)
    
    sys.stdout.write('\n')

    ''' For opencv video creation '''
    #print(OUTPUT_FRAME_LIST)
    print(SIM_FACE_OBJ_ID)
    output_video = cv2.VideoWriter("./output.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), int(vcap.get(cv2.CAP_PROP_FPS)), (int(STREAMMUX_WIDTH), int(STREAMMUX_HEIGHT)))
    output_frame_idx = 0
    for frame_idx in range(int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = vcap.read()
        if not ret:
            break
        if output_frame_idx == len(OUTPUT_FRAME_LIST):
            break
        
        if frame_idx == OUTPUT_FRAME_LIST[output_frame_idx]:
            for bbox in OUTPUT_BBOX_LIST[output_frame_idx]:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 3)
            output_video.write(frame)
            output_frame_idx += 1   

    vcap.release()
    output_video.release()
    ''' End '''


def parse_args():
    global SOURCE, CONFIG_INFER_PGIE, STREAMMUX_BATCH_SIZE, STREAMMUX_WIDTH, STREAMMUX_HEIGHT, GPU_ID, \
        PERF_MEASUREMENT_INTERVAL_SEC

    parser = argparse.ArgumentParser(description='DeepStream')
    parser.add_argument('-s', '--source', required=True, help='Source stream/file')
    parser.add_argument('-b', '--streammux-batch-size', type=int, default=1, help='Streammux batch-size (default: 1)')
    parser.add_argument('-w', '--streammux-width', type=int, default=1920, help='Streammux width (default: 1920)')
    parser.add_argument('-e', '--streammux-height', type=int, default=1080, help='Streammux height (default: 1080)')
    parser.add_argument('-g', '--gpu-id', type=int, default=0, help='GPU id (default: 0)')
    parser.add_argument('-f', '--fps-interval', type=int, default=5, help='FPS measurement interval (default: 5)')
    args = parser.parse_args()
    if args.source == '':
        sys.stderr.write('ERROR: Source not found\n')
        sys.exit(1)

    SOURCE = args.source
    STREAMMUX_BATCH_SIZE = args.streammux_batch_size
    STREAMMUX_WIDTH = args.streammux_width
    STREAMMUX_HEIGHT = args.streammux_height
    GPU_ID = args.gpu_id
    PERF_MEASUREMENT_INTERVAL_SEC = args.fps_interval


if __name__ == '__main__':
    parse_args()
    get_face_pool()
    sys.exit(summarize())
