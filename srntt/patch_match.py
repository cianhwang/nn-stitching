# Copyright by SRNTT.
# Usage: M_t, M_s = Fun_patchMatching(M_LR, M_LRef, M_Ref, patchSize, Stride)
# Goal: mapping stack of concerned patch to low-res feature map
# Input: Feature Map M_LR, M_LRef, M_Ref, 40 x 40 x 256 in size. patchSize and Stride are 3 and 1 by default.
# Output: M_t, 40 x 40 x 256.
# Structure: 
    # def Fun_patchSample(M_LRef, Stride) return patchStack
        # zero_padding: YES
        # each patch: 3 x 3 x 256
        # patch stack: 3 x 3 x 256 x (40x40)
        # auxiliary functions: fun_zeroPadding, fun_patchCrop

    # def Fun_patchConv(Patch, M_LR, ~Neighborhood) return scoreMap
        # score Map: 40 x 40 x 256 x (40*40)
        ## ~Neighborhood: patch-related area in M_LR.

    # def Fun_locateCorr(scoreMap) return M_s, maxIdxMap
        # M_s 40 x 40 x 256: containing max score
        # maxIdxMap: 40x40

    # def Fun_stickPatch(maxIdxMap, M_Ref, M_s) return M_t
        # M_t 40 x 40 x 256

# Future Improvement:
    # search similar patch in surrounding area. (Fun_patchConv)

import tensorflow as tf
import numpy as np
#from numba import jit
import time

def fun_zeroPadding(M, fringe):
    m, n, band = M.shape
    M_padding = np.zeros([m+fringe*2, n+fringe*2, band])
    M_padding[fringe:-fringe, fringe:-fringe, :] = M
    return M_padding


def fun_patchCrop(M, leftupperX, leftupperY, patchSize): #
    # M No.x40x40x256
    patch = M[leftupperX:(leftupperX+patchSize), leftupperY:(leftupperY+patchSize), :]
    return patch

#def fun_simScore(patch_LR, patch_LRef):
    vec_LR = patch_LR.reshape([-1, 1])
    vec_LRef = patch_LRef.reshape([-1, 1])
    score = np.dot(vec_LR.T, vec_LRef)/(np.linalg.norm(vec_LR)*np.linalg.norm(vec_LRef)+ np.spacing(1))
    return score


def Fun_patchSample(M, patchSize = 3, Stride = 1):
    m, n, band = M.shape
    assert m == n
    assert m % Stride == 0
    assert (m-patchSize)/Stride+1 > 0

    patchStack = np.zeros([patchSize, patchSize, band, int(m*n/Stride**2)])
    M_padding = fun_zeroPadding(M, int((patchSize-1)/2))
    k = 0
    for i in range(0, m, Stride):
        for j in range(0, n, Stride):
            singlePatch = fun_patchCrop(M_padding, i, j, patchSize)
            patchStack[:,:,:,k] = singlePatch#/(np.spacing(1)+np.linalg.norm(singlePatch.reshape([1, -1]))) # Normalization
            k += 1
    return patchStack

def Fun_patchConv(M_LR, patchStack, sess):

    m1, n1, band1 = M_LR.shape
    m2, n2, band2, dim = patchStack.shape # m2: patch size
    assert band1 == band2

    M_LR_patchStack = Fun_patchSample(M_LR)
    normMap = np.zeros([m1, n1])
    k = 0
    for i in range(m1): # Norm map
        for j in range(n1):
            normMap[i, j] = np.linalg.norm(M_LR_patchStack[:, :, :, k].reshape([1, -1]))
            k += 1
    k = 0
    for i in range(m1): # Norm map
        for j in range(n1):
            patchStack[:, :, :, k] = patchStack[:, :, :, k]/(np.spacing(1)+np.linalg.norm(patchStack[:, :, :, k].reshape([1, -1])))
            k += 1

# -----conv: use similarity score.-----
    tf_patchStack = tf.constant(patchStack, tf.float32)
    tf_M_LR = tf.constant(M_LR[np.newaxis, :, :, :], tf.float32)
    scoreMap = tf.nn.conv2d(tf_M_LR, tf_patchStack, strides = [1,1,1,1], padding='SAME') # 1.Inner product? 2.Flip?
    scoreMap = sess.run(scoreMap)
    
    scoreMap = scoreMap[0, :, :, :]/(np.spacing(1)+np.repeat(normMap[:, :, np.newaxis], dim, axis=2))
    return scoreMap


def Fun_locateCorr(scoreMap, band):
    assert len(scoreMap.shape) == 3
    M_s = np.amax(scoreMap, axis = 2)
    M_s = np.repeat(M_s[:, :, np.newaxis], band, axis=2) ##? Normalization?
    maxIdxMap = np.argmax(scoreMap, axis = 2)
    return M_s, maxIdxMap


def Fun_stickPatch(maxIdxMap, M_Ref, M_s, patchSize = 3):
    m, n, band = M_Ref.shape
    stickPad = np.zeros([m + patchSize-1, n + patchSize-1, band])
    patchStack = Fun_patchSample(M_Ref, patchSize)
    fringe = int((patchSize-1)/2)
    for i in range(m):
        for j in range(n):
            patchIdx = maxIdxMap[i, j]
            maxPatch = patchStack[:,:,:, patchIdx]
            assert len(maxPatch.shape) == 3
            if i != 0 and j != 0 and i != m-1 and j != n-1: # interior, patch leftupper coordinate = (i, j)
                stickPad[i:(i+patchSize), j:(j+patchSize), :] = maxPatch/9
            elif (i == 0 and j == 0) or (i == 0 and j == n) \
                or (i == m and j == 0) or (i == m and j == n): # 4 cornor
                stickPad[i:(i+patchSize), j:(j+patchSize), :] = maxPatch/4
            else: # rim
                stickPad[i:(i+patchSize), j:(j+patchSize), :] = maxPatch/6

    M_t = stickPad[fringe:-fringe, fringe:-fringe, :] * M_s # element-wise product
    return M_t


def Fun_patchMatching(M_LR, M_LRef, M_Ref, sess, patchSize = 3, Stride = 1):
    M_s = np.zeros(M_LR.shape)
    M_t = np.zeros(M_LR.shape)
    for i in range(M_LR.shape[0]):
        M_LRef_patchStack = Fun_patchSample(M_LRef[i,:,:,:], patchSize, Stride)
        scoreMap = Fun_patchConv(M_LR[i,:,:,:], M_LRef_patchStack, sess)
        M_s[i,:,:,:], maxIdxMap = Fun_locateCorr(scoreMap, M_LR[i,:,:,:].shape[2])
        M_t[i,:,:,:] = Fun_stickPatch(maxIdxMap, M_Ref[i,:,:,:], M_s[i,:,:,:], patchSize)
    return M_t, M_s

def test():
    sess = tf.Session()
    M_LR = (np.random.random([10, 40, 40, 256])-0.5)*8000
    M_LRef = (np.random.random([10, 40, 40, 256])-0.5)*8000
    M_Ref = (np.random.random([10, 40, 40, 256])-0.5)*8000
    startTime = time.time()
    Fun_patchMatching(M_LR, M_LRef, M_Ref, sess)
    print("time =", time.time()-startTime)

if __name__ == "__main__":
    test()
