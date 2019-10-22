import traceback
from enum import IntEnum
import cv2
import numpy as np
from pathlib import Path

from utils import Path_utils
from utils.DFLPNG import DFLPNG
from utils.DFLJPG import DFLJPG

from .Sample import Sample
from .Sample import SampleType

from facelib import FaceType
from facelib import LandmarksProcessor
from interact import interact as io

class SampleLoader:
    cache = dict()

    @staticmethod
    def load(sample_type, samples_path, target_samples_path=None):
        cache = SampleLoader.cache

        if str(samples_path) not in cache.keys():
            cache[str(samples_path)] = [None]*SampleType.QTY

        datas = cache[str(samples_path)]

        if            sample_type == SampleType.IMAGE:
            if  datas[sample_type] is None:
                datas[sample_type] = [ Sample(filename=filename) for filename in io.progress_bar_generator( Path_utils.get_image_paths(samples_path), "Loading") ]

        elif          sample_type == SampleType.FACE:
            if  datas[sample_type] is None:
                datas[sample_type] = SampleLoader.upgradeToFaceSamples( [ Sample(filename=filename) for filename in Path_utils.get_image_paths(samples_path) ] )

        elif          sample_type == SampleType.FACE_YAW_SORTED:
            if  datas[sample_type] is None:
                datas[sample_type] = SampleLoader.upgradeToFaceYawSortedSamples( SampleLoader.load(SampleType.FACE, samples_path) )

        elif          sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
            if  datas[sample_type] is None:
                if target_samples_path is None:
                    raise Exception('target_samples_path is None for FACE_YAW_SORTED_AS_TARGET')
                datas[sample_type] = SampleLoader.upgradeToFaceYawSortedAsTargetSamples( SampleLoader.load(SampleType.FACE_YAW_SORTED, samples_path), SampleLoader.load(SampleType.FACE_YAW_SORTED, target_samples_path) )
        elif          sample_type == SampleType.FACE_WITH_CLOSE_TO_SELF:
            if  datas[sample_type] is None:
                datas[sample_type] = SampleLoader.upgradeToFaceCloseToSelfSamples( SampleLoader.load(SampleType.FACE, samples_path) )


        return datas[sample_type]

    @staticmethod
    def upgradeToFaceSamples ( samples ):
        sample_list = []

        for s in io.progress_bar_generator(samples, "Loading"):
            s_filename_path = Path(s.filename)
            try:
                if s_filename_path.suffix == '.png':
                    dflimg = DFLPNG.load ( str(s_filename_path) )
                elif s_filename_path.suffix == '.jpg':
                    dflimg = DFLJPG.load ( str(s_filename_path) )
                else:
                    dflimg = None

                if dflimg is None:
                    print ("%s is not a dfl image file required for training" % (s_filename_path.name) )
                    continue

                pitch, yaw = LandmarksProcessor.estimate_pitch_yaw ( dflimg.get_landmarks() )

                sample_list.append( s.copy_and_set(sample_type=SampleType.FACE,
                                                   face_type=FaceType.fromString (dflimg.get_face_type()),
                                                   shape=dflimg.get_shape(),
                                                   landmarks=dflimg.get_landmarks(),
                                                   pitch=pitch,
                                                   yaw=yaw) )
            except:
                print ("Unable to load %s , error: %s" % (str(s_filename_path), traceback.format_exc() ) )

        return sample_list

    @staticmethod
    def upgradeToFaceCloseToSelfSamples (samples):
        yaw_samples = SampleLoader.upgradeToFaceYawSortedSamples(samples)
        yaw_samples_len = len(yaw_samples)

        sample_list = []
        for i in io.progress_bar_generator( range(yaw_samples_len), "Sorting"):
            if yaw_samples[i] is not None:
                for s in yaw_samples[i]:
                    s_t = []

                    for n in range(2000):
                        yaw_idx = np.clip ( i-10 +np.random.randint(20), 0, yaw_samples_len-1 )
                        if yaw_samples[yaw_idx] is None:
                            continue

                        yaw_idx_samples_len = len(yaw_samples[yaw_idx])

                        yaw_idx_sample = yaw_samples[yaw_idx][ np.random.randint(yaw_idx_samples_len) ]
                        if s.filename == yaw_idx_sample.filename:
                            continue

                        s_t.append ( yaw_idx_sample )
                        if len(s_t) >= 50:
                            break

                    if len(s_t) == 0:
                        s_t = [s]

                    sample_list.append( s.copy_and_set(close_target_list = s_t) )

        return sample_list

    @staticmethod
    def upgradeToFaceYawSortedSamples( samples ):

        lowest_yaw, highest_yaw = -1.0, 1.0
        gradations = 64
        diff_rot_per_grad = abs(highest_yaw-lowest_yaw) / gradations

        yaws_sample_list = [None]*gradations

        for i in io.progress_bar_generator(range(gradations), "Sorting"):
            yaw = lowest_yaw + i*diff_rot_per_grad
            next_yaw = lowest_yaw + (i+1)*diff_rot_per_grad

            yaw_samples = []
            for s in samples:
                s_yaw = s.yaw
                if (i == 0            and s_yaw < next_yaw) or \
                   (i  < gradations-1 and s_yaw >= yaw and s_yaw < next_yaw) or \
                   (i == gradations-1 and s_yaw >= yaw):
                    yaw_samples.append ( s.copy_and_set(sample_type=SampleType.FACE_YAW_SORTED) )

            if len(yaw_samples) > 0:
                yaws_sample_list[i] = yaw_samples

        return yaws_sample_list

    @staticmethod
    def upgradeToFaceYawSortedAsTargetSamples (s, t):
        l = len(s)
        if l != len(t):
            raise Exception('upgradeToFaceYawSortedAsTargetSamples() s_len != t_len')
        b = l // 2

        s_idxs = np.argwhere ( np.array ( [ 1 if x != None else 0  for x in s] ) == 1 )[:,0]
        t_idxs = np.argwhere ( np.array ( [ 1 if x != None else 0  for x in t] ) == 1 )[:,0]

        new_s = [None]*l

        for t_idx in t_idxs:
            search_idxs = []
            for i in range(0,l):
                search_idxs += [t_idx - i, (l-t_idx-1) - i, t_idx + i, (l-t_idx-1) + i]

            for search_idx in search_idxs:
                if search_idx in s_idxs:
                    mirrored = ( t_idx != search_idx and ((t_idx < b and search_idx >= b) or (search_idx < b and t_idx >= b)) )
                    new_s[t_idx] = [ sample.copy_and_set(sample_type=SampleType.FACE_YAW_SORTED_AS_TARGET,
                                                         mirror=True,
                                                         yaw=-sample.yaw,
                                                         landmarks=LandmarksProcessor.mirror_landmarks (sample.landmarks, sample.shape[1] ))
                                          for sample in s[search_idx]
                                        ] if mirrored else s[search_idx]
                    break

        return new_s
