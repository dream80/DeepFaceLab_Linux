import traceback
import numpy as np
import random
import cv2
import multiprocessing
from utils import iter_utils
from facelib import LandmarksProcessor

from samples import SampleType
from samples import SampleProcessor
from samples import SampleLoader
from samples import SampleGeneratorBase

'''
arg
output_sample_types = [ 
                        [SampleProcessor.TypeFlags, size, (optional)random_sub_size] ,
                        ...
                      ]
'''
class SampleGeneratorFace(SampleGeneratorBase):
    def __init__ (self, samples_path, debug, batch_size, sort_by_yaw=False, sort_by_yaw_target_samples_path=None, with_close_to_self=False, sample_process_options=SampleProcessor.Options(), output_sample_types=[], add_sample_idx=False, add_pitch=False, add_yaw=False, generators_count=2, **kwargs):
        super().__init__(samples_path, debug, batch_size)
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        self.add_sample_idx = add_sample_idx
        self.add_pitch = add_pitch
        self.add_yaw = add_yaw

        if sort_by_yaw_target_samples_path is not None:
            self.sample_type = SampleType.FACE_YAW_SORTED_AS_TARGET
        elif sort_by_yaw:
            self.sample_type = SampleType.FACE_YAW_SORTED
        elif with_close_to_self:
            self.sample_type = SampleType.FACE_WITH_CLOSE_TO_SELF
        else:
            self.sample_type = SampleType.FACE

        self.samples = SampleLoader.load (self.sample_type, self.samples_path, sort_by_yaw_target_samples_path)

        if self.debug:
            self.generators_count = 1
            self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, 0 )]
        else:
            self.generators_count = min ( generators_count, len(self.samples) )
            self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, i ) for i in range(self.generators_count) ]

        self.generators_sq = [ multiprocessing.Queue() for _ in range(self.generators_count) ]

        self.generator_counter = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    #forces to repeat these sample idxs as fast as possible
    #currently unused
    def repeat_sample_idxs(self, idxs): # [ idx, ... ]
        #send idxs list to all sub generators.
        for gen_sq in self.generators_sq:
            gen_sq.put (idxs)

    def batch_func(self, generator_id):
        gen_sq = self.generators_sq[generator_id]
        samples = self.samples
        samples_len = len(samples)
        samples_idxs = [ *range(samples_len) ] [generator_id::self.generators_count]
        repeat_samples_idxs = []

        if len(samples_idxs) == 0:
            raise ValueError('No training data provided.')

        if self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
            if all ( [ samples[idx] == None for idx in samples_idxs] ):
                raise ValueError('Not enough training data. Gather more faces!')

        if self.sample_type == SampleType.FACE or self.sample_type == SampleType.FACE_WITH_CLOSE_TO_SELF:
            shuffle_idxs = []
        elif self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
            shuffle_idxs = []
            shuffle_idxs_2D = [[]]*samples_len

        while True:
            while not gen_sq.empty():
                idxs = gen_sq.get()
                for idx in idxs:
                    if idx in samples_idxs:
                        repeat_samples_idxs.append(idx)

            batches = None
            for n_batch in range(self.batch_size):
                while True:
                    sample = None

                    if len(repeat_samples_idxs) > 0:
                        idx = repeat_samples_idxs.pop()
                        if self.sample_type == SampleType.FACE or self.sample_type == SampleType.FACE_WITH_CLOSE_TO_SELF:
                            sample = samples[idx]
                        elif self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
                            sample = samples[(idx >> 16) & 0xFFFF][idx & 0xFFFF]
                    else:
                        if self.sample_type == SampleType.FACE or self.sample_type == SampleType.FACE_WITH_CLOSE_TO_SELF:
                            if len(shuffle_idxs) == 0:
                                shuffle_idxs = samples_idxs.copy()
                                np.random.shuffle(shuffle_idxs)

                            idx = shuffle_idxs.pop()
                            sample = samples[ idx ]

                        elif self.sample_type == SampleType.FACE_YAW_SORTED or self.sample_type == SampleType.FACE_YAW_SORTED_AS_TARGET:
                            if len(shuffle_idxs) == 0:
                                shuffle_idxs = samples_idxs.copy()
                                np.random.shuffle(shuffle_idxs)

                            idx = shuffle_idxs.pop()
                            if samples[idx] != None:
                                if len(shuffle_idxs_2D[idx]) == 0:
                                    shuffle_idxs_2D[idx] = random.sample( range(len(samples[idx])), len(samples[idx]) )

                                idx2 = shuffle_idxs_2D[idx].pop()
                                sample = samples[idx][idx2]

                                idx = (idx << 16) | (idx2 & 0xFFFF)

                    if sample is not None:
                        try:
                            x = SampleProcessor.process (sample, self.sample_process_options, self.output_sample_types, self.debug)
                        except:
                            raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                        if type(x) != tuple and type(x) != list:
                            raise Exception('SampleProcessor.process returns NOT tuple/list')

                        if batches is None:
                            batches = [ [] for _ in range(len(x)) ]
                            if self.add_sample_idx:
                                batches += [ [] ]
                                i_sample_idx = len(batches)-1
                            if self.add_pitch:
                                batches += [ [] ]
                                i_pitch = len(batches)-1
                            if self.add_yaw:
                                batches += [ [] ]
                                i_yaw = len(batches)-1

                        for i in range(len(x)):
                            batches[i].append ( x[i] )

                        if self.add_sample_idx:
                            batches[i_sample_idx].append (idx)

                        if self.add_pitch or self.add_yaw:
                            pitch, yaw = LandmarksProcessor.estimate_pitch_yaw (sample.landmarks)

                        if self.add_pitch:
                            batches[i_pitch].append ([pitch])

                        if self.add_yaw:
                            batches[i_yaw].append ([yaw])

                        break
            yield [ np.array(batch) for batch in batches]
