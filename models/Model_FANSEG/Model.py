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
    def onInitializeOptions(self, is_first_run, ask_override):            
        default_face_type = 'f'
        if is_first_run:
            self.options['face_type'] = io.input_str ("Half or Full face? (h/f, ?:help skip:f) : ", default_face_type, ['h','f'], help_message="Half face has better resolution, but covers less area of cheeks.").lower()
        else:
            self.options['face_type'] = self.options.get('face_type', default_face_type)
     
    #override
    def onInitialize(self):
        exec(nnlib.import_all(), locals(), globals())
        self.set_vram_batch_requirements( {1.5:4} )

        self.resolution = 256
        self.face_type = FaceType.FULL if self.options['face_type'] == 'f' else FaceType.HALF

        
        self.fan_seg = FANSegmentator(self.resolution, 
                                      FaceType.toString(self.face_type), 
                                      load_weights=not self.is_first_run(),
                                      weights_file_root=self.get_model_root_path(),
                                      training=True)

        if self.is_training_mode:
            f = SampleProcessor.TypeFlags
            face_type = f.FACE_TYPE_FULL if self.options['face_type'] == 'f' else f.FACE_TYPE_HALF
            
            self.set_training_data_generators ([    
                    SampleGeneratorFace(self.training_data_src_path, debug=self.is_debug(), batch_size=self.batch_size, 
                            sample_process_options=SampleProcessor.Options(random_flip=True, motion_blur = [25, 1] ), 
                            output_sample_types=[ [f.WARPED_TRANSFORMED | face_type | f.MODE_BGR_SHUFFLE | f.OPT_APPLY_MOTION_BLUR, self.resolution],
                                                  [f.WARPED_TRANSFORMED | face_type | f.MODE_M | f.FACE_MASK_FULL, self.resolution]
                                                ]),
                                                
                    SampleGeneratorFace(self.training_data_dst_path, debug=self.is_debug(), batch_size=self.batch_size, 
                            sample_process_options=SampleProcessor.Options(random_flip=True ), 
                            output_sample_types=[ [f.TRANSFORMED | face_type | f.MODE_BGR_SHUFFLE, self.resolution]
                                                ])
                                               ])
                
    #override
    def onSave(self):        
        self.fan_seg.save_weights()
        
    #override
    def onTrainOneIter(self, generators_samples, generators_list):
        target_src, target_src_mask = generators_samples[0]

        loss,acc = self.fan_seg.train_on_batch( [target_src], [target_src_mask] )

        return ( ('loss', loss), ('acc',acc))
        
    #override
    def onGetPreview(self, sample):
        test_A   = sample[0][0][0:4] #first 4 samples
        test_B   = sample[1][0][0:4] #first 4 samples
        
        mAA = self.fan_seg.extract(test_A)
        mBB = self.fan_seg.extract(test_B)

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
                
        return [ ('training data', np.concatenate ( st, axis=0 ) ),
                 ('evaluating data', np.concatenate ( st2, axis=0 ) ),
                 ]
