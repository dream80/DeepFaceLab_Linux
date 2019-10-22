import numpy as np

from nnlib import nnlib
from models import ModelBase
from facelib import FaceType
from facelib import FANSegmentator
from samplelib import *
from interact import interact as io

class Model(ModelBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, 
                            ask_write_preview_history=False, 
                            ask_target_iter=False,
                            ask_sort_by_yaw=False,
                            ask_random_flip=False,
                            ask_src_scale_mod=False)
        
    #override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {1.5:4} )

        self.resolution = 256
        self.face_type = FaceType.FULL
        
        self.fan_seg = FANSegmentator(self.resolution, 
                                      FaceType.toString(self.face_type), 
                                      load_weights=not self.is_first_run(),
                                      weights_file_root=self.get_model_root_path(),
                                      training=True)

        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            f_type = f.FACE_TYPE_FULL
            
            self.set_training_data_generators ([    
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, 
                            sample_process_options=SampleProcessor.Options(random_flip=True, motion_blur = [25, 1], normalize_tanh = True ), 
                            output_sample_types=[ [f.TRANSFORMED | f_type | f.MODE_BGR_SHUFFLE | f.OPT_APPLY_MOTION_BLUR, self.resolution],
                                                  [f.TRANSFORMED | f_type | f.MODE_M | f.FACE_MASK_FULL, self.resolution]
                                                ]),
                                                
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                            sample_process_options=SampleProcessor.Options(random_flip=True, normalize_tanh = True ), 
                            output_sample_types=[ [f.TRANSFORMED | f_type | f.MODE_BGR_SHUFFLE, self.resolution]
                                                ])
                                               ])
                
    #override
    def onSave(self):        
        self.fan_seg.save_weights()
        
    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        target_src, target_src_mask = generators_samples[0]

        loss = self.fan_seg.train_on_batch( [target_src], [target_src_mask] )

        return ( ('loss', loss), )
        
    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][0][0:4] #first 4 samples
        test_B   = sample[1][0][0:4] #first 4 samples
        
        mAA = self.fan_seg.extract_from_bgr([test_A])
        mBB = self.fan_seg.extract_from_bgr([test_B])
        
        test_A, test_B, = [ np.clip( (x + 1.0)/2.0, 0.0, 1.0)  for x in [test_A, test_B] ]
        
        mAA = np.repeat ( mAA, (3,), -1)
        mBB = np.repeat ( mBB, (3,), -1)
        
        st = []
        for i in range(0, len(test_A)):
            st.append ( np.concatenate ( (
                test_A[i,:,:,0:3],
                mAA[i],
                test_A[i,:,:,0:3]*mAA[i],
                ), axis=1) )
                
        st2 = []
        for i in range(0, len(test_B)):
            st2.append ( np.concatenate ( (
                test_B[i,:,:,0:3],
                mBB[i],
                test_B[i,:,:,0:3]*mBB[i],
                ), axis=1) )
                
        return [ ('FANSegmentator', np.concatenate ( st, axis=0 ) ),
                 ('never seen', np.concatenate ( st2, axis=0 ) ),
                 ]
