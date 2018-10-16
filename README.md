# SRNTT Project Realization
This is a Tensorflow implemention of SRNTT. (Zhifei Zhang, Zhaowen Wang, Zhe Lin, and Hairong Qi, "Reference-Conditioned Super-Resolution by Neural Texture Transfer", arXiv:1804.03360v1, 2018.)

# OFFLINE BRANCH OF SRNTT
## Module 1: Texture Extractor (''test_vgg19.py'')

**Description:** pre-trained VGG19 model.

**Function:** vgg19_pretrained()

**Input:** image, 160 x 160 x 3

**Output:** feature matrix of conv3_1 (40x40x256)

feature matrix of conv5_1 (10x10x512)

## Module 2: Patch Matching & Texture Swapping (''patch_match.py'')

**Description:** swap low-res image with high-res patches.

**Function:** Fun_patchMatching()

**Input:** M_LR, M_LRef, M_Ref (40x40x256)

**Output:** M_t, M_s (40x40x256)
