#!/usr/bin/python

# Description: Use this class to perform:
# 1. Mask to box/rrect/ellipse functions
# 2. Image Augmentation for an input file directory (rotation, etc)
# 3. Splitting a training set into positive/negative areas if masks exist
# 4. Finding minimum distances for series of images

import glob
import math
import cv2
import re
from numpy import *
from scipy import misc
import os
import scipy.spatial.distance
import itertools
import numpy as np
import collections
import glob

class Image_Utils:

	def order_pics(self,indir,regex="",type="tif"):
		# Generates an output csv which contains the predicted order of images and their distance to 
		# the next image
		image_files = glob.iglob(indir+"/"+regex+"*."+type)

		cnt=0
		images = dict()
		for f in image_files:
			img = cv2.imread(f)
			img = misc.imresize(img,[128,128])
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			images[f] = img
			cnt += 1
		print("Info: Read a total of "+str(cnt)+" images")
		features = { f : self.image_features(images[f]) for f in images }
		print("Info: Found features for each image!")
		dists    = self.feature_dists(features)
		print("Info: Found distances for all images!")
		f_seq    = self.image_sequence(dists)
		print("Info: Determined sequence for all images..")
		return [f_seq, dists]

	def image_features(self,img):
    		return self.tile_features(img)   # a tile is just an image...

	def tile_features(self,tile, tile_min_side = 50):
    		# Recursively split a tile (image) into quadrants, down to a minimum 
    		# tile size, then return flat array of the mean brightness in those tiles.
    		tile_x, tile_y = tile.shape
    		mid_x = tile_x / 2
    		mid_y = tile_y / 2
    		if (mid_x < tile_min_side) or (mid_y < tile_min_side):
        		return np.array([tile.mean()]) # hit minimum tile size
    		else:
        		tiles = [ tile[:mid_x, :mid_y ], tile[mid_x:, :mid_y ], 
                  		  tile[:mid_x , mid_y:], tile[mid_x:,  mid_y:] ] 
        		features = [self.tile_features(t) for t in tiles]
        	return np.array(features).flatten()

	def feature_dist(self,feats_0, feats_1):
    		# Definition of the distance metric between image features
    		return scipy.spatial.distance.euclidean(feats_0, feats_1)


	def feature_dists(self,features):
    		# Calculate the distance between all pairs of images (using their features)
    		dists = collections.defaultdict(dict)
    		f_img_features = features.keys()
    		for f_img0, f_img1 in itertools.permutations(f_img_features, 2):
        		dists[f_img0][f_img1] = self.feature_dist(features[f_img0], features[f_img1])
    		return dists

	def image_sequence(self,dists):
    		# Return a sequence of images that minimizes the sum of 
    		# inter-image distances. This function relies on image_seq_start(), 
    		# which requires an arbitray starting image. 
    		# In order to find an even lower-cost sequence, this function
    		# tries all possible staring images and returns the best result.

    		f_starts = dists.keys()
    		seqs = [self.image_seq_start(dists, f_start) for f_start in f_starts]
		if len(seqs) > 0:
	    		dist_best, seq_best = list(sorted(seqs))[0]
    			return seq_best
		else:
			return None

	def image_seq_start(self, dists, f_start):

    		# Given a starting image (i.e. named f_start), greedily pick a sequence 
    		# of nearest-neighbor images until there are no more unpicked images. 

    		f_picked = [f_start]
    		f_unpicked = set(dists.keys()) - set([f_start])
    		f_current = f_start
    		dist_tot = 0

    		while f_unpicked:
        		# Collect the distances from the current image to the 
        		# remaining unpicked images, then pick the nearest one 
        		candidates = [(dists[f_current][f_next], f_next) for f_next in f_unpicked]
        		dist_nearest, f_nearest = list(sorted(candidates))[0]
        		# Update the image accounting & make the nearest image the current image 
        		f_unpicked.remove(f_nearest)
        		f_picked.append(f_nearest)
        		dist_tot += dist_nearest
        		f_current = f_nearest 
    		return (dist_tot, f_picked)


	# splits an image directory into positive and negative 
	# useful for the case of say the ultrasound/nerve kaggle competition
	def split_image_dir(self,picdir,maskdir,filetype="jpg",
			    out_picdir_pos="./pics_pos",out_picdir_neg="./pics_neg",
			    out_maskdir_pos="./masks_pos",out_maskdir_neg="./masks_neg",
			    verbose=True):
		if not os.path.isdir(out_picdir_pos):
			print("Info: Creating directory "+out_picdir_pos)
			os.system("mkdir -p "+out_picdir_pos)
                if not os.path.isdir(out_picdir_neg):
                        print("Info: Creating directory "+out_picdir_neg)
                        os.system("mkdir -p "+out_picdir_neg)
                if not os.path.isdir(out_maskdir_pos):
                        print("Info: Creating directory "+out_maskdir_pos)
                        os.system("mkdir -p "+out_maskdir_pos)
                if not os.path.isdir(out_maskdir_neg):
                        print("Info: Creating directory "+out_maskdir_neg)
                        os.system("mkdir -p "+out_maskdir_neg)

		masks = glob.iglob(maskdir+"/*"+filetype)
		for mask in masks:
			img = cv2.imread(mask)
                        searchObj= re.search(r'([^\/\\]+)\.'+filetype,mask)
			pic_name = re.sub('_mask',"",searchObj.group(1))
			if len(where(img > 0)[0]) > 0:
				if verbose is True:
					print("Info: Pic/Mask "+mask+" is positive, splitting to "+out_maskdir_pos)
				os.system("cp "+mask+" "+out_maskdir_pos)
				os.system("cp "+picdir+"/"+pic_name+"."+filetype+" "+out_picdir_pos)
			else:
				if verbose is True:	
					print("Info: Pic/Mask "+mask+" is negative, splitting to "+out_maskdir_neg)
				os.system("cp "+mask+" "+out_maskdir_neg)
				os.system("cp "+picdir+"/"+pic_name+"."+filetype+" "+out_picdir_neg)

	# augments an image directory by switches specified
	def augment_image_dir(self,indir,outdir,filetype="jpg",
			      rotate=True,rotate_max=5,rotate_crop=30,
			      flip_x=True,flip_x_rotate=True,
			      warp_left=False,warp_right=False,
		              warp_top=False,warp_bot=False,
			      warp_max=50,
			      copy_orig=True,
			      verbose=True):
		pics = glob.iglob(indir+"/*"+filetype)
		orig_cnt=0
		aug_cnt=0

		if not os.path.isdir(outdir):
			os.system("mkdir -p "+outdir)

		for pic in pics:
			img = cv2.imread(pic)
			height = shape(img)[0]
			width  = shape(img)[1]
			orig_cnt += 1
			searchObj= re.search(r'([^\/\\\.]+)\.'+filetype,pic)

			if copy_orig is True:
				os.system("cp "+str(pic)+" "+outdir)
				if verbose is True:
					print("Info: Copied original "+str(pic)+" to "+outdir)
                                aug_cnt += 1

			if flip_x is True:
				img_mx = fliplr(img)
				cv2.imwrite(outdir+"/"+searchObj.group(1)+".mx."+filetype,img_mx)
				if verbose is True:
					print("Info: Generated xflip aug to "+outdir+"/"+searchObj.group(1)+".mx."+filetype)
				aug_cnt += 1

                                if flip_x_rotate is True:
                                        img_rp = self.rotate_image(img_mx,rotate_max)
                                        img_rm = self.rotate_image(img_mx,-rotate_max)

                                        img_rp = misc.imresize(img_rp[rotate_crop:shape(img_rp)[0]-rotate_crop,rotate_crop:shape(img_rp)[1]-rotate_crop],[shape(img)[0],shape(img)[1]])
                                        img_rm = misc.imresize(img_rm[rotate_crop:shape(img_rm)[0]-rotate_crop,rotate_crop:shape(img_rm)[1]-rotate_crop],[shape(img)[0],shape(img)[1]])

                                        cv2.imwrite(outdir+"/"+searchObj.group(1)+".mx_cw."+filetype,img_rp)
                                        cv2.imwrite(outdir+"/"+searchObj.group(1)+".mx_ccw."+filetype,img_rm)
                                        if verbose is True:
                                                print("Info: Generated flipped, x-rotated +"+str(rotate_max)+" aug to "+outdir+"/"+searchObj.group(1)+".mx_cw."+filetype)
                                                print("Info: Generated flipped, x-rotated -"+str(rotate_max)+" aug to "+outdir+"/"+searchObj.group(1)+".mx_ccw."+filetype)
                                        aug_cnt += 2

			dst = array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype=float32)	
		
			if warp_top is True:
				src = array([[warp_max,0],[width-warp_max-1,0],[width-1,height-1],[0,height-1]],dtype=float32)
				M = cv2.getPerspectiveTransform(src,dst)
				warped_img = cv2.warpPerspective(img, M, (width,height))
				cv2.imwrite(outdir+"/"+searchObj.group(1)+".wtop."+filetype,warped_img)
				if verbose is True:
					print("Info: Generated warped top aug to "+outdir+"/"+searchObj.group(1)+".wtop."+filetype)
				aug_cnt += 1

                        if warp_bot is True:
                                src = array([[0,0],[width-1,0],[width-warp_max-1,height-1],[warp_max,height-1]],dtype=float32)
                                M = cv2.getPerspectiveTransform(src,dst)
                                warped_img = cv2.warpPerspective(img, M, (width,height))
                                cv2.imwrite(outdir+"/"+searchObj.group(1)+".wbot."+filetype,warped_img)
				if verbose is True:
					print("Info: Generated warped bot aug to "+outdir+"/"+searchObj.group(1)+".wbot."+filetype)
				aug_cnt += 1

	                if warp_left is True:
                                src = array([[0,warp_max],[width-1,0],[width-1,height-1],[0,height-warp_max-1]],dtype=float32)
                                M = cv2.getPerspectiveTransform(src,dst)
                                warped_img = cv2.warpPerspective(img, M, (width,height))
                                cv2.imwrite(outdir+"/"+searchObj.group(1)+".wleft."+filetype,warped_img)
                                if verbose is True:
                                        print("Info: Generated warped left aug to "+outdir+"/"+searchObj.group(1)+".wleft."+filetype)
				aug_cnt += 1

                        if warp_right is True:
                                src = array([[0,0],[width-1,warp_max],[width-1,height-warp_right-1],[0,height-1]],dtype=float32)
                                M = cv2.getPerspectiveTransform(src,dst)
                                warped_img = cv2.warpPerspective(img, M, (width,height))
                                cv2.imwrite(outdir+"/"+searchObj.group(1)+".wright."+filetype,warped_img)
                                if verbose is True:
                                        print("Info: Generated warped right aug to "+outdir+"/"+searchObj.group(1)+".wright."+filetype)
				aug_cnt += 1

			if rotate is True:
				img_rp = self.rotate_image(img,rotate_max)
				img_rm = self.rotate_image(img,-rotate_max)

				img_rp = misc.imresize(img_rp[rotate_crop:shape(img_rp)[0]-rotate_crop,rotate_crop:shape(img_rp)[1]-rotate_crop],[shape(img)[0],shape(img)[1]])
				img_rm = misc.imresize(img_rm[rotate_crop:shape(img_rm)[0]-rotate_crop,rotate_crop:shape(img_rm)[1]-rotate_crop],[shape(img)[0],shape(img)[1]])

				cv2.imwrite(outdir+"/"+searchObj.group(1)+".cw."+filetype,img_rp)
				cv2.imwrite(outdir+"/"+searchObj.group(1)+".ccw."+filetype,img_rm)
				if verbose is True:
					print("Info: Generated rotated +"+str(rotate_max)+" aug to "+outdir+"/"+searchObj.group(1)+".cw."+filetype)
					print("Info: Generated rotated -"+str(rotate_max)+" aug to "+outdir+"/"+searchObj.group(1)+".ccw."+filetype)
				aug_cnt += 2
		print("Info: Original image count="+str(orig_cnt))
		print("Info: Augmented image count="+str(aug_cnt))

		
	# rotates matrix mat by angle angle (degrees)
	# negative angles accepted
	def rotate_image(self, mat, angle):
    		height, width = mat.shape[:2]
    		image_center = (width / 2, height / 2)

    		rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    		radians = math.radians(angle)
    		sin = math.sin(radians)
    		cos = math.cos(radians)
    		bound_w = int((height * abs(sin)) + (width * abs(cos)))
    		bound_h = int((height * abs(cos)) + (width * abs(sin)))

    		rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    		rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    		rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    		return rotated_mat

	# where m = num of samples
	# takes an array of m x h x w mask
	# and return a m x 4 where 4 corresponds to x1,y1,x2,y2
	# bounding boxes
        def mask2box(self, Y_mask):
		num_samples = shape(Y_mask)[0]
		height = shape(Y_mask)[1]
		width  = shape(Y_mask)[2]
		Y_bbox = zeros(shape=(num_samples,4))

		for i in range(0,num_samples):
        		xyc = where(Y_mask[i] != 0)
        		y1 = min(xyc[0])
        		x1 = min(xyc[1])
		        y2 = max(xyc[0])
        		x2 = max(xyc[1])
        		Y_bbox[i] = [x1,y1,x2,y2]
		# scale boxes to -1,1
		Y_bbox = Y_bbox.astype(float32)
		Y_bbox[:,0] = (Y_bbox[:,0] - height)/height
		Y_bbox[:,2] = (Y_bbox[:,2] - height)/height
		Y_bbox[:,1] = (Y_bbox[:,1] - width)/width
		Y_bbox[:,3] = (Y_bbox[:,3] - width)/width
		return Y_bbox

	# same as mask2box, except returns ellipse
	# x,y,w,h, and angle of ellipse
	def mask2ellipse(self, Y_mask):
                num_samples = shape(Y_mask)[0]
                height = shape(Y_mask)[1]
                width  = shape(Y_mask)[2]
                Y_ellipse = zeros(shape=(num_samples,5))
		return Y_ellipse
