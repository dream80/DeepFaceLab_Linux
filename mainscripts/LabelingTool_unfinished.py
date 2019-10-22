import traceback
import os
import sys
import time
import numpy as np
import numpy.linalg as npl

import cv2
from pathlib import Path
from interact import interact as io
from utils.cv2_utils import *
from utils import Path_utils
from utils.DFLPNG import DFLPNG
from utils.DFLJPG import DFLJPG
from facelib import LandmarksProcessor

def main(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise ValueError('Input directory not found. Please ensure it exists.')

    if not output_path.exists():
        output_path.mkdir(parents=True)

    wnd_name = "Labeling tool"
    io.named_window (wnd_name)
    io.capture_mouse(wnd_name)
    io.capture_keys(wnd_name)

    #for filename in io.progress_bar_generator (Path_utils.get_image_paths(input_path), desc="Labeling"):
    for filename in Path_utils.get_image_paths(input_path):
        filepath = Path(filename)

        if filepath.suffix == '.png':
            dflimg = DFLPNG.load( str(filepath) )
        elif filepath.suffix == '.jpg':
            dflimg = DFLJPG.load ( str(filepath) )
        else:
            dflimg = None

        if dflimg is None:
            io.log_err ("%s is not a dfl image file" % (filepath.name) )
            continue

        lmrks = dflimg.get_landmarks()
        lmrks_list = lmrks.tolist()
        orig_img = cv2_imread(str(filepath))
        h,w,c = orig_img.shape

        mask_orig = LandmarksProcessor.get_image_hull_mask( orig_img.shape, lmrks).astype(np.uint8)[:,:,0]
        ero_dil_rate = w // 8
        mask_ero = cv2.erode (mask_orig, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero_dil_rate,ero_dil_rate)), iterations = 1 )
        mask_dil = cv2.dilate(mask_orig, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ero_dil_rate,ero_dil_rate)), iterations = 1 )


        #mask_bg = np.zeros(orig_img.shape[:2],np.uint8)
        mask_bg = 1-mask_dil
        mask_bgp = np.ones(orig_img.shape[:2],np.uint8) #default - all background possible
        mask_fg = np.zeros(orig_img.shape[:2],np.uint8)
        mask_fgp = np.zeros(orig_img.shape[:2],np.uint8)

        img = orig_img.copy()

        l_thick=2

        def draw_4_lines (masks_out, pts, thickness=1):
            fgp,fg,bg,bgp = masks_out
            h,w = fg.shape

            fgp_pts = []
            fg_pts = np.array([ pts[i:i+2]  for i in range(len(pts)-1)])
            bg_pts = []
            bgp_pts = []

            for i in range(len(fg_pts)):
                a, b = line = fg_pts[i]

                ba = b-a
                v = ba / npl.norm(ba)

                ccpv = np.array([v[1],-v[0]])
                cpv = np.array([-v[1],v[0]])
                step = 1 / max(np.abs(cpv))

                fgp_pts.append ( np.clip (line + ccpv * step * thickness,     0, w-1 ).astype(np.int) )
                bg_pts.append  ( np.clip (line +  cpv * step * thickness,     0, w-1 ).astype(np.int) )
                bgp_pts.append ( np.clip (line +  cpv * step * thickness * 2, 0, w-1 ).astype(np.int) )

            fgp_pts = np.array(fgp_pts)
            bg_pts = np.array(bg_pts)
            bgp_pts = np.array(bgp_pts)

            cv2.polylines(fgp, fgp_pts, False, (1,), thickness=thickness)
            cv2.polylines(fg,  fg_pts,  False, (1,), thickness=thickness)
            cv2.polylines(bg,  bg_pts,  False, (1,), thickness=thickness)
            cv2.polylines(bgp, bgp_pts, False, (1,), thickness=thickness)

        def draw_lines ( masks_steps, pts, thickness=1):
            lines = np.array([ pts[i:i+2] for i in range(len(pts)-1)])


            for mask, step in masks_steps:
                h,w = mask.shape

                mask_lines = []
                for i in range(len(lines)):
                    a, b = line = lines[i]
                    ba = b-a
                    ba_len = npl.norm(ba)
                    if ba_len != 0:
                        v = ba / ba_len
                        pv = np.array([-v[1],v[0]])
                        pv_inv_max = 1 / max(np.abs(pv))
                        mask_lines.append ( np.clip (line + pv * pv_inv_max * thickness * step, 0, w-1 ).astype(np.int) )
                    else:
                        mask_lines.append ( np.array(line, dtype=np.int) )
                cv2.polylines(mask, mask_lines, False, (1,), thickness=thickness)

        def draw_fill_convex( mask_out, pts, scale=1.0 ):
            hull = cv2.convexHull(np.array(pts))

            if scale !=1.0:
                pts_count = hull.shape[0]

                sum_x = np.sum(hull[:, 0, 0])
                sum_y = np.sum(hull[:, 0, 1])

                hull_center = np.array([sum_x/pts_count, sum_y/pts_count])
                hull = hull_center+(hull-hull_center)*scale
                hull = hull.astype(pts.dtype)
            cv2.fillConvexPoly( mask_out, hull, (1,) )

        def get_gc_mask_bgr(gc_mask):
            h, w = gc_mask.shape
            bgr = np.zeros( (h,w,3), dtype=np.uint8 )

            bgr [ gc_mask == 0 ] = (0,0,0)
            bgr [ gc_mask == 1 ] = (255,255,255)
            bgr [ gc_mask == 2 ] = (0,0,255) #RED
            bgr [ gc_mask == 3 ] = (0,255,0) #GREEN
            return bgr

        def get_gc_mask_result(gc_mask):
            return np.where((gc_mask==1) + (gc_mask==3),1,0).astype(np.int)

        #convex inner of right chin to end of right eyebrow
        #draw_fill_convex ( mask_fgp, lmrks_list[8:17]+lmrks_list[26:27] )

        #convex inner of start right chin to right eyebrow
        #draw_fill_convex ( mask_fgp, lmrks_list[8:9]+lmrks_list[22:27] )

        #convex inner of nose
        draw_fill_convex ( mask_fgp, lmrks[27:36] )

        #convex inner of nose half
        draw_fill_convex ( mask_fg, lmrks[27:36], scale=0.5 )


        #left corner of mouth to left corner of nose
        #draw_lines ( [ (mask_fg,0),   ], lmrks_list[49:50]+lmrks_list[32:33], l_thick)

        #convex inner: right corner of nose to centers of eyebrows
        #draw_fill_convex ( mask_fgp, lmrks_list[35:36]+lmrks_list[19:20]+lmrks_list[24:25])

        #right corner of mouth to right corner of nose
        #draw_lines ( [ (mask_fg,0),   ], lmrks_list[54:55]+lmrks_list[35:36], l_thick)

        #left eye
        #draw_fill_convex ( mask_fg, lmrks_list[36:40] )
        #right eye
        #draw_fill_convex ( mask_fg, lmrks_list[42:48] )

        #right chin
        draw_lines ( [ (mask_bg,0), (mask_fg,-1),   ], lmrks[8:17], l_thick)

        #left eyebrow center to right eyeprow center
        draw_lines ( [ (mask_bg,-1), (mask_fg,0),   ], lmrks_list[19:20] + lmrks_list[24:25], l_thick)
        #        #draw_lines ( [ (mask_bg,-1), (mask_fg,0),   ], lmrks_list[24:25] + lmrks_list[19:17:-1], l_thick)

        #half right eyebrow to end of right chin
        draw_lines ( [ (mask_bg,-1), (mask_fg,0),   ], lmrks_list[24:27] + lmrks_list[16:17], l_thick)

        #import code
        #code.interact(local=dict(globals(), **locals()))

        #compose mask layers
        gc_mask = np.zeros(orig_img.shape[:2],np.uint8)
        gc_mask [ mask_bgp==1 ] = 2
        gc_mask [ mask_fgp==1 ] = 3
        gc_mask [ mask_bg==1  ] = 0
        gc_mask [ mask_fg==1  ] = 1

        gc_bgr_before = get_gc_mask_bgr (gc_mask)



        #io.show_image (wnd_name, gc_mask )

        ##points, hierarcy = cv2.findContours(original_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        ##gc_mask = ( (1-erode_mask)*2 + erode_mask )# * dilate_mask
        #gc_mask = (1-erode_mask)*2 + erode_mask
        #cv2.addWeighted(
        #gc_mask = mask_0_27 + (1-mask_0_27)*2
        #
        ##import code
        ##code.interact(local=dict(globals(), **locals()))
        #
        #rect = (1,1,img.shape[1]-2,img.shape[0]-2)
        #
        #
        cv2.grabCut(img,gc_mask,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),5, cv2.GC_INIT_WITH_MASK)

        gc_bgr = get_gc_mask_bgr (gc_mask)
        gc_mask_result = get_gc_mask_result(gc_mask)
        gc_mask_result_1 = gc_mask_result[:,:,np.newaxis]

        #import code
        #code.interact(local=dict(globals(), **locals()))
        orig_img_gc_layers_masked = (0.5*orig_img + 0.5*gc_bgr).astype(np.uint8)
        orig_img_gc_before_layers_masked = (0.5*orig_img + 0.5*gc_bgr_before).astype(np.uint8)



        pink_bg = np.full ( orig_img.shape, (255,0,255), dtype=np.uint8 )


        orig_img_result = orig_img * gc_mask_result_1
        orig_img_result_pinked = orig_img_result + pink_bg * (1-gc_mask_result_1)

        #io.show_image (wnd_name, blended_img)

        ##gc_mask, bgdModel, fgdModel =
        #
        #mask2 = np.where((gc_mask==1) + (gc_mask==3),255,0).astype('uint8')[:,:,np.newaxis]
        #mask2 = np.repeat(mask2, (3,), -1)
        #
        ##mask2 = np.where(gc_mask!=0,255,0).astype('uint8')
        #blended_img = orig_img #-\
        #              #0.3 * np.full(original_img.shape, (50,50,50)) * (1-mask_0_27)[:,:,np.newaxis]
        #              #0.3 * np.full(original_img.shape, (50,50,50)) * (1-dilate_mask)[:,:,np.newaxis] +\
        #              #0.3 * np.full(original_img.shape, (50,50,50)) * (erode_mask)[:,:,np.newaxis]
        #blended_img = np.clip(blended_img, 0, 255).astype(np.uint8)
        ##import code
        ##code.interact(local=dict(globals(), **locals()))
        orig_img_lmrked = orig_img.copy()
        LandmarksProcessor.draw_landmarks(orig_img_lmrked, lmrks, transparent_mask=True)

        screen = np.concatenate ([orig_img_gc_before_layers_masked,
                                  orig_img_gc_layers_masked,
                                  orig_img,
                                  orig_img_lmrked,
                                  orig_img_result_pinked,
                                  orig_img_result,
                                  ], axis=1)

        io.show_image (wnd_name, screen.astype(np.uint8) )


        while True:
            io.process_messages()

            for (x,y,ev,flags) in io.get_mouse_events(wnd_name):
                pass
                #print (x,y,ev,flags)

            key_events = [ ev for ev, in io.get_key_events(wnd_name) ]
            for key in key_events:
                if key == ord('1'):
                    pass
                if key == ord('2'):
                    pass
                if key == ord('3'):
                    pass

            if ord(' ') in key_events:
                break

    import code
    code.interact(local=dict(globals(), **locals()))




#original_mask = np.ones(original_img.shape[:2],np.uint8)*2
#cv2.drawContours(original_mask, points, -1, (1,), 1)
