import numpy as np
import cv2

class Pyramid(object):
    @classmethod
    def numlevels(cls,im_sz):
        min_d=min(im_sz)
        nlev=1
        while min_d>1:
            nlev=nlev+1
            min_d=(min_d+1)//2
        print('Pyramid depth: %d'%(nlev-1))
        return nlev-1
    @classmethod
    def gaussian(cls, img, num_levels=None):
        if num_levels is None:
            num_levels = cls.numlevels(img.shape)
        lower = img.copy()
        gaussian_pyr = [lower]
        for i in range(num_levels):
            lower = cv2.pyrDown(lower)
            gaussian_pyr.append(lower.astype('float32'))
        return gaussian_pyr
    @classmethod
    def laplacian_from(cls, gaussian_pyr):
        laplacian_top = gaussian_pyr[-1]
        num_levels = len(gaussian_pyr) - 1
        laplacian_pyr = [laplacian_top.astype('float32')]
        for i in range(num_levels,0,-1):
            size = (gaussian_pyr[i-1].shape[1], gaussian_pyr[i-1].shape[0])
            gaussian_expanded = cv2.pyrUp(gaussian_pyr[i], dstsize=size)
            laplacian = np.subtract(gaussian_pyr[i-1], gaussian_expanded)
            laplacian_pyr.append(laplacian.astype('float32'))
        return laplacian_pyr
    @classmethod
    def laplacian(cls, img, num_levels=None):
        gp = cls.gaussian(img,num_levels)
        return cls.laplacian_from(gp)
    @classmethod
    def reconstruct(cls, laplacian_pyr):
        laplacian_top = laplacian_pyr[0]
        rec = laplacian_top
        num_levels = len(laplacian_pyr) - 1
        for i in range(num_levels):
            size = (laplacian_pyr[i + 1].shape[1], laplacian_pyr[i + 1].shape[0])
            laplacian_expanded = cv2.pyrUp(rec, dstsize=size).astype('float32')
            rec = cv2.add(laplacian_pyr[i+1].astype('float32'), laplacian_expanded)
        return rec