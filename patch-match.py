# Copyright by SRNTT.
# Goal: mapping stack of concerned patch to low-res feature map
# Input: Feature Map M_LR, M_LRef, M_Ref, 40 x 40 x 30 in size.
# Output: M_t, 40 x 40 x 3.
# Structure: 
    # def fun_imgRead()

    # def fun_patchSample(M_LRef, Stride) return patchStack
        # each patch: 3 x 3
        # patch stack: 3 x 3 x (40x40) zero_padding: YES
    # def fun_patchConv(Patch, M_LR, ~Neighborhood) return score_Map, ~maxIdx_Map
        # score Map: 40 x 40 x (40*40)
        ## ~maxIdx_Map: 40 x 40 x 1
        ## Neighborhood: patch-related area in M_LR.
        # consider in-place storage. Memory-friendly.
    # def fun_locateCorr(score_Map) return M_s
        # spot max point slice
        # M_s 40x40
    # def fun_combine(M_s, M_Ref) return M_t
        # multiplication, blending(average)
        # M_t 40x40
    # def fun_simScore(patch_LR, patch_LRef) return score 
# Future Improvement:
    # search similar patch in surrounding area.

import tensorflow as tf
import numpy as np

def fun_patchSample(M, Stride = 1):
    m, n, band = M.shape
    assert (m==n and band==3)
    patchSize = 3
    patchStack = np.zeros([m, n, band, m*n])
    M_padding = tf.image.pad_to_bounding(M, 1, 1, m+2, n+2)
    for i in range(m):
        for j in range(n):
            singlePatch = fun_patchCrop(
                M_padding, i, j, patchSize
            )
            patchStack[:,:,:,i*m+j] = singlePatch
    return patchStack

def fun_patchCrop():
    pass

def fun_patchConv(M_LR, patchStack):
    m1, n1, band1 = M_LR.shape
    m2, n2, band2 = patchStack.shape
    assert band1 == band2
    # !!-----conv: use similarity score.-----
    # tf_patchStack = tf.constant(patchStack, tf.float32)
    # tf_M_LR = tf.constant(M_LR[np.newaxis, :, :, :], tf.float32)
    # scoreMap = tf.nn.conv2d(tf_M_LR, tf_patchStack, strides = [1,1,1,1], padding='SAME')
    # with tf.Session() as sess:
    #     scoreMap = sess.run(scoreMap)

    return scoreMap

def fun_simScore(patch_LR, patch_LRef):
    vec_LR = patch_LR.reshape([-1, 1])
    vec_LRef = patch_LRef.reshape([-1, 1])
    score = np.dot(vec_LR.T, vec_LRef)/(np.linalg.norm(vec_LR)*np.linalg.norm(vec_LRef))

def fun_locateCorr(scoreMap):
    maxMap = np.amax(scoreMap, axis=2) #np.amax
    maxIdxMap = np.where(scoreMap==maxMap) # !!need refine.

def fun_combine(M_s, M_Ref):
    m, n, band = M_s.shape()
    M_t = np.zeros(m, n, band)
    patchStack = fun_patchSample(M_Ref)
    for i in range(m):
        for j in range(m):
            patchIdx = M_s[i, j]
            maxPatch = patchStack[:,:,:, patchIdx]
            M_t = fun_stickPatch(M_t, maxPatch) 
            # CAUTION: for overlapped area, set mean.
    return M_t

def fun_stickPatch(M_t, maxPatch):
    pass

    




