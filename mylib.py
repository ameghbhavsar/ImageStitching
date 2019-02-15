# Instructions:
# Do not change the output file names, use the helper functions as you see fit

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import imutils

def feature_matching(img1, img2, savefig=False):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches2to1 = flann.knnMatch(des2,des1,k=2)

    matchesMask_ratio = [[0,0] for i in xrange(len(matches2to1))]
    match_dict = {}
    for i,(m,n) in enumerate(matches2to1):
        if m.distance < 0.7*n.distance:
            matchesMask_ratio[i]=[1,0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1,des2,k=2)
    matchesMask_ratio_recip = [[0,0] for i in xrange(len(recip_matches))]

    for i,(m,n) in enumerate(recip_matches):
        if m.distance < 0.7*n.distance: # ratio
            if m.queryIdx in match_dict and match_dict[m.queryIdx] == m.trainIdx: #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i]=[1,0]



    if savefig:
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = (255,0,0),
                           matchesMask = matchesMask_ratio_recip,
                           flags = 0)
        img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,recip_matches,None,**draw_params)

        plt.figure(),plt.xticks([]),plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png",bbox_inches='tight')

    return ([ kp1[m.queryIdx].pt for m in good ],[ kp2[m.trainIdx].pt for m in good ])

def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0,0]

    im_h,im_w = img1.shape[:2]

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h,cyl_w = cyl.shape[:2]
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0,cyl_w):
        for y_cyl in np.arange(0,cyl_h):
            theta = (x_cyl - x_c) / f
            h     = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K,X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl),int(x_cyl)] = img1[int(y_im),int(x_im)]
            cyl_mask[int(y_cyl),int(x_cyl)] = 255


    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png",bbox_inches='tight')

    return (cyl,cyl_mask)


def getTransform(src, dst, method='affine'):
    pts1,pts2 = feature_matching(src,dst)

    src_pts = np.float32(pts1).reshape(-1,1,2)
    dst_pts = np.float32(pts2).reshape(-1,1,2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        #M = np.append(M, [[0,0,1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)


def Laplacian_blending(img1,img2,mask,levels=4):

    G1 = img1.copy()
    G2 = img2.copy()
    GM = mask.copy()
    # print(G1.shape, G2.shape)
    gp1 = [G1]
    gp2 = [G2]
    gpM = [GM]
    for i in xrange(levels):
        G1 = cv2.pyrDown(G1)
        # print(G1.shape)
        G2 = cv2.pyrDown(G2)
        GM = cv2.pyrDown(GM)
        gp1.append(np.float32(G1))
        gp2.append(np.float32(G2))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lp1  = [gp1[levels-1]] # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lp2  = [gp2[levels-1]]
    gpMr = [gpM[levels-1]]
    for i in xrange(levels-1,0,-1):
    # Laplacian: subtarct upscaled version of lower level from current level
    # to get the high frequencies
        L1 = np.subtract(gp1[i-1], cv2.pyrUp(gp1[i], dstsize=(gp1[i-1].shape[1], gp1[i-1].shape[0])))
        L2 = np.subtract(gp2[i-1], cv2.pyrUp(gp2[i], dstsize=(gp2[i-1].shape[1], gp2[i-1].shape[0])))
        lp1.append(L1)
        lp2.append(L2)
        gpMr.append(gpM[i-1]) # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for l1,l2,gm in zip(lp1,lp2,gpMr):
        ls = l1 * gm + l2 * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    # print(LS.shape)
    ls_ = LS[0]
    print(ls_.shape)
    for i in xrange(1,levels):
        ls_ = cv2.pyrUp(ls_, dstsize=(gp1[levels-i-1].shape[1], gp1[levels-i-1].shape[0]))
        ls_ = cv2.add(ls_, LS[i])

    return ls_

def cylindrical_warping(img1, img2, flag):

    m1 = np.ones_like(img2, dtype='float32')
    h,w = m1.shape[:2]
    f = 400
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    m1,maskB = cylindricalWarpImage(m1, K)
    m1 = cv2.copyMakeBorder(m1,50,50,300,300, cv2.BORDER_CONSTANT)

    if not flag:
        h,w = img1.shape[:2]
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
        img1,mask1 = cylindricalWarpImage(img1, K)
        img1 = cv2.copyMakeBorder(img1,50,50,600,600, cv2.BORDER_CONSTANT)


    h,w = img2.shape[:2]
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    img2,mask2 = cylindricalWarpImage(img2, K)
    img2 = cv2.copyMakeBorder(img2,50,200,50,400, cv2.BORDER_CONSTANT)

    (M1, pts3, pts4, mask6) = getTransform(img2, img1,'affine')

    out1 = cv2.warpAffine(img2, M1, (img1.shape[1],img1.shape[0]))
    out4 = cv2.warpAffine(m1, M1, (img1.shape[1],img1.shape[0]))


    output = Laplacian_blending(out1,img1,out4,3)

    cv2.imwrite('output_cylindrical_lpb.png',output)

    o=cv2.imread('output_cylindrical_lpb.png',1)

    return o
