from __future__ import print_function
import os, sys, math, argparse
import glob
import cv2
import numpy as np
import reconstruction as recon

IMGPATTERN = "*.png"


def readFileList(imgFolder):
    imgFileList = glob.glob(os.path.join(imgFolder, IMGPATTERN))
    # self.imgFileList = os.listdir(self.imgFolder)
    # self.imgFileList.remove('.DS_Store') # remove system database log
    imgFileList.sort()

    return imgFileList

def processBoth(imgPath):

    imgFileList = readFileList(imgPath)
    resultPath = os.path.join(imgPath, "results")
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    assert (len(imgFileList) == 12), ("Images in the directory should be 12. Now is ",len(imgFileList))

    for i in range(len(imgFileList)):

        img = cv2.imread(os.path.join(imgPath, imgFileList[i]), cv2.IMREAD_COLOR)

        if i == 0:
            imgSetDeflectometry = np.zeros((8, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
            imgSetGradientIllu = np.zeros((4, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

        if i < 8:
            imgSetDeflectometry[i, ...] = img
        else:
            imgSetGradientIllu[i -8, ...] = img

    deflectometry = recon.Deflectometry(imgSetDeflectometry)
    deflectometry.setNormal()
    deflectometry.exportNormal(resultPath)
    MeshD = recon.Mesh("deflectometry", deflectometry.height, deflectometry.width)
    MeshD.setNormal(deflectometry.normal)
    MeshD.setTexture(deflectometry.albedo)
    MeshD.setDepth(1)
    MeshD.exportOBJ(resultPath, True)

    gradientillu = recon.GradientIllumination(imgSetGradientIllu)
    gradientillu.setNormal()
    gradientillu.exportNormal(resultPath)
    MeshGI = recon.Mesh("gradientillumination", gradientillu.height, gradientillu.width)
    MeshGI.setNormal(gradientillu.normal)
    MeshGI.setTexture(gradientillu.albedo)
    MeshGI.setDepth(1)
    MeshGI.exportOBJ(resultPath, True)

def processDeflectometry(imgPath):

    imgFileList = readFileList(imgPath)
    resultPath = os.path.join(imgPath, "results")
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    assert (len(imgFileList) == 8), ("Images in the directory should be 8. Now is ",len(imgFileList))

    for i in range(len(imgFileList)):

        img = cv2.imread(os.path.join(imgPath, imgFileList[i]), cv2.IMREAD_COLOR)

        if i == 0:
            imgSetDeflectometry = np.zeros((8, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

        imgSetDeflectometry[i, ...] = img


    deflectometry = recon.Deflectometry(imgSetDeflectometry)
    deflectometry.setNormal()
    deflectometry.exportNormal(resultPath)
    MeshD = recon.Mesh("deflectometry", deflectometry.height, deflectometry.width)
    MeshD.setNormal(deflectometry.normal)
    MeshD.setTexture(deflectometry.albedo)
    MeshD.setDepth(1)
    MeshD.exportOBJ(resultPath, True)

def processGI(imgPath):

    imgFileList = readFileList(imgPath)
    resultPath = os.path.join(imgPath, "results")
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)

    assert (len(imgFileList) == 4), ("Images in the directory should be 4. Now is ",len(imgFileList))

    for i in range(len(imgFileList)):

        img = cv2.imread(os.path.join(imgPath, imgFileList[i]), cv2.IMREAD_COLOR)

        if i == 0:
            imgSetGradientIllu = np.zeros((4, img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

        imgSetGradientIllu[i, ...] = img

    gradientillu = recon.GradientIllumination(imgSetGradientIllu)
    gradientillu.setNormal()
    gradientillu.exportNormal(resultPath)
    MeshGI = recon.Mesh("gradientillumination", gradientillu.height, gradientillu.width)
    MeshGI.setNormal(gradientillu.normal)
    MeshGI.setTexture(gradientillu.albedo)
    MeshGI.setDepth(1)
    MeshGI.exportOBJ(resultPath, True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=int, help="Mode. 0:Both, 1:Deflectometry, 2:Gradient Illumination ")
    parser.add_argument("-d", "--directory", type=str, help="Path of images directory")
    args = parser.parse_args()

    if args.directory is not None:
        imgPath = args.directory
    if args.mode is not None:
        mode = args.mode

    assert mode <= 2, ("Can't recognize the mode!")

    if mode == 0 :
        processBoth(imgPath)
    elif mode == 1:
        processDeflectometry(imgPath)
    elif mode == 2:
        processGI(imgPath)