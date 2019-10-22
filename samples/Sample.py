from enum import IntEnum
import cv2
import numpy as np
from utils.cv2_utils import *

class SampleType(IntEnum):
    IMAGE = 0 #raw image

    FACE_BEGIN = 1
    FACE = 1                      #aligned face unsorted
    FACE_YAW_SORTED = 2           #sorted by yaw
    FACE_YAW_SORTED_AS_TARGET = 3 #sorted by yaw and included only yaws which exist in TARGET also automatic mirrored
    FACE_WITH_CLOSE_TO_SELF = 4
    FACE_END = 4

    QTY = 5

class Sample(object):
    def __init__(self, sample_type=None, filename=None, face_type=None, shape=None, landmarks=None, pitch=None, yaw=None, mirror=None, close_target_list=None):
        self.sample_type = sample_type if sample_type is not None else SampleType.IMAGE
        self.filename = filename
        self.face_type = face_type
        self.shape = shape
        self.landmarks = np.array(landmarks) if landmarks is not None else None
        self.pitch = pitch
        self.yaw = yaw
        self.mirror = mirror
        self.close_target_list = close_target_list

    def copy_and_set(self, sample_type=None, filename=None, face_type=None, shape=None, landmarks=None, pitch=None, yaw=None, mirror=None, close_target_list=None):
        return Sample(
            sample_type=sample_type if sample_type is not None else self.sample_type,
            filename=filename if filename is not None else self.filename,
            face_type=face_type if face_type is not None else self.face_type,
            shape=shape if shape is not None else self.shape,
            landmarks=landmarks if landmarks is not None else self.landmarks.copy(),
            pitch=pitch if pitch is not None else self.pitch,
            yaw=yaw if yaw is not None else self.yaw,
            mirror=mirror if mirror is not None else self.mirror,
            close_target_list=close_target_list if close_target_list is not None else self.close_target_list)

    def load_bgr(self):
        img = cv2_imread (self.filename).astype(np.float32) / 255.0
        if self.mirror:
            img = img[:,::-1].copy()
        return img

    def get_random_close_target_sample(self):
        if self.close_target_list is None:
            return None
        return self.close_target_list[randint (0, len(self.close_target_list)-1)]
