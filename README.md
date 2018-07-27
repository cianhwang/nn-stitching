# OFFLINE BRANCH OF SRNTT
## Module 1: Texture Extractor

Description: pre-trained VGG19 model.

Function: vgg19_pretrained()

Input: image, 160 x 160 x 3

Output: feature matrix of conv3_1 (40x40x256)

        and 

        feature matrix of conv5_1 (10x10x512)

## Module 2: Patch Matching & Texture Swapping

Description: swap low-res image with high-res patches.

Function: Fun_patchMatching()

Input: M_LR, M_LRef, M_Ref (40x40x256)

Output: M_t, M_s (40x40x256)