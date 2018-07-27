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

def fun_zeroPadding(M, fringe):
    m, n, band = M.shape
    M_padding = np.zeros([m+fringe*2, n+fringe*2, band])
    M_padding[fringe:-fringe, fringe:-fringe, :] = M
    return M_padding

def fun_patchCrop(M, leftupperX, leftupperY, patchSize): #
    # M 40x40x256
    patch = M[leftupperX:(leftupperX+patchSize), leftupperY:(leftupperY+patchSize), :]
    return patch

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
            patchStack[:,:,:,k] = singlePatch/(np.spacing(1)+np.linalg.norm(singlePatch.reshape([1, -1]))) # Normalization
            k += 1
    return patchStack

def Fun_patchConv(M_LR, patchStack):
    m1, n1, band1 = M_LR.shape
    m2, n2, band2, dim = patchStack.shape # m2: patch size
    assert band1 == band2
    # -----conv: use similarity score.-----
    tf_patchStack = tf.constant(patchStack, tf.float32)
    tf_M_LR = tf.constant(M_LR[np.newaxis, :, :, :], tf.float32)
    scoreMap = tf.nn.conv2d(tf_M_LR, tf_patchStack, strides = [1,1,1,1], padding='SAME') # 1.Inner product? 2.Flip?

    with tf.Session() as sess:
        scoreMap = sess.run(scoreMap)
    scoreMap = scoreMap[0, :, :, :]
    return scoreMap
#
    # M_LR_zeroPadding = fun_zeroPadding(M_LR, (m2-1)/2)
    # scoreMap = np.zeros([m1, n1])
    # maxIdxMap = np.zeros([m1, n1])
    # for i in range(m1):
    #     for j in range(n1):
    #         for k in range(dim): # optimize...
    #             patch_M_LR = fun_patchCrop(M_LR, i, j, m2)
    #             simScore = fun_simScore(patch, patch_M_LR)
    #             if simScore > scoreMap[i, j]:
    #                 maxIdxMap[i, j] = k
    #                 scoreMap[i, j] = simScore
    # return scoreMap, maxIdxMap

# def fun_simScore(patch_LR, patch_LRef):
    # vec_LR = patch_LR.reshape([-1, 1])
    # vec_LRef = patch_LRef.reshape([-1, 1])
    # score = np.dot(vec_LR.T, vec_LRef)/(np.linalg.norm(vec_LR)*np.linalg.norm(vec_LRef))

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

def Fun_patchMatching(M_LR, M_LRef, M_Ref, patchSize = 3, Stride = 1):
    M_LRef_patchStack = Fun_patchSample(M_LRef, patchSize, Stride)
    M_Ref_patchStack = Fun_patchSample(M_Ref, patchSize, Stride)
    scoreMap = Fun_patchConv(M_LR, M_LRef_patchStack)
    M_s, maxIdxMap = Fun_locateCorr(scoreMap, M_LR.shape[2])
    M_t = Fun_stickPatch(maxIdxMap, M_Ref, M_s, patchSize)
    return M_t, M_s


M_LR = np.random.random([40, 40, 256])
M_LRef = np.random.random([40, 40, 256])
M_Ref = np.random.random([40, 40, 256])
M_s, M_t = Fun_patchMatching(M_LR, M_LRef, M_Ref)


# ''' test program '''
# M_LR = np.array([0, 1, 0, 0, 0])
# M_LR = np.tile(M_LR, (5, 1))
# M_LR = M_LR[:, :, np.newaxis]
# M_LRef = np.zeros([5, 5, 1])
# smPatch = np.zeros([3, 3])
# smPatch[:, 0] = -1
# smPatch[:, 2] = 1
# M_LRef[0:3, 0:3, 0] = smPatch

# M_Ref = M_LRef

# M_s, M_t = Fun_patchMatching(M_LR, M_LRef, M_Ref)
# print(M_s)
# print(M_t)

