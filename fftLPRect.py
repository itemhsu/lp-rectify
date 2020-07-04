#depends on
#pip install  Pillow
#pip install  sklearn
#pip install  opencv-python
#pip install  matplotlib

import os, sys
import numpy as np
import math
from PIL import Image
import PIL
import cv2
isBias=False
isPlot=False


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def imgWrapA(orgImg,a):
    column=orgImg.shape[1]
    row=orgImg.shape[0]
    pts1 = np.float32([[column/2,row/2],[column/2,row/4],[column/4,row/2-column/4*a]])
    pts2 = np.float32([[column/2,row/2],[column/2,row/4],[column/4,row/2]])
    M = cv2.getAffineTransform(pts1,pts2)
    imgWarpAffine = cv2.warpAffine(orgImg,M,(column,row))
    return imgWarpAffine

def rectScale(im):
    myWidth = im.shape[0]
    myHeight = int(myWidth/2)
    nim = cv2.resize(im, (myWidth*2, myHeight*2), interpolation=cv2.INTER_AREA)
    return nim

def estCorrect(orgImg0, cutoffF=0.8, margin=0.1):
    orgImg = cv2.fastNlMeansDenoisingColored(orgImg0,None,9,9,7,21)
    w=orgImg.shape[1]
    crop_img = orgImg[:, int(margin*w):int(w-margin*w)]
    pix_color = np.array(crop_img)
    full_pix_color = np.array(orgImg)
    img = rgb2gray(pix_color)
    img=abs(img[0:-1,:]-img[1:,:])
    
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
        
    #plt.subplot(143),plt.imshow(np.abs(fshift), cmap = 'gray')
    #plt.title('np.abs(fshift)'), plt.xticks([]), plt.yticks([])
    margin = 0.9 # Cut off the outer 10% of the image
    # Do the polar rotation along 100 angular steps with a radius of 256 pixels.
    size=min(img.shape)
    polar_img = cv2.warpPolar(magnitude_spectrum, (int(size/2), 200), (img.shape[1]/2,img.shape[0]/2), 
                                  size*margin*0.5, cv2.WARP_POLAR_LINEAR)
   
    polar_img_lowF=polar_img[:,int(0.01*polar_img.shape[1]):int(cutoffF*polar_img.shape[1])]
    
    polar_sum_200=np.sum(polar_img_lowF,axis=1)
    polar_sum=polar_sum_200[0:100]+polar_sum_200[100:200]
    #polar_sum[50]=min(polar_sum) #matthew  do not count center line
    #print(statistics.stdev(polar_sum[25:75]))
    maxIndex=np.argmax(polar_sum[25:75])+25
    offsetDegree=(maxIndex-50)/100*3.14
    aEst=np.sin(offsetDegree)
    full_pix_color0 = np.array(orgImg0)
    correctImg=imgWrapA(full_pix_color0,aEst)
    #full_pix_color
    #polar_sum[25:75]=min(polar_sum) #matthew  do not count center line
    #maxIndex2=np.argmax(polar_sum)
    #print("maxIndex2={}".format(maxIndex2))

    return correctImg

def estCorrect2D(orgImg, cutoffF=0.8, margin=0.1):
    hCorrectedImg=estCorrect(orgImg, cutoffF, margin)
    hCorrectedImg90=np.rot90(hCorrectedImg, )
    vCorrectedImg=estCorrect(hCorrectedImg90, cutoffF, margin)
    vCorrectedImg270=np.rot90(vCorrectedImg,  k=3)

    
    return hCorrectedImg, vCorrectedImg270


def estCorrect2D_scale(strFpath, cutoffF=0.8, margin=0.1):
    orgImg = cv2.imread(strFpath)
    pix_color = np.array(orgImg)
    hCorrectedImg, vCorrectedImg270=estCorrect2D(pix_color, cutoffF, margin)
    imWrap_t_wrap_t_scaled=rectScale(vCorrectedImg270)
    return hCorrectedImg, vCorrectedImg270, imWrap_t_wrap_t_scaled
    

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    imWrap_s=[]
    imWrap_t_wrap_t_s=[]
    imWrap_t_wrap_t_scaled_s=[]
    cutoffF=0.8
    margin=0.1
    dirs={'img','img3', 'img4', 'img5'}
    for fDir in dirs:
        for root,unkown,fNames in os.walk(fDir):
            for f in fNames:
                strFpath="{}/{}".format(fDir,f)
                print(strFpath)
                hCorrectedImg, vCorrectedImg270, imWrap_t_wrap_t_scaled=estCorrect2D_scale(strFpath, cutoffF, margin)
                imWrap_s.append(hCorrectedImg)
                imWrap_t_wrap_t_s.append(vCorrectedImg270)
                imWrap_t_wrap_t_scaled_s.append(imWrap_t_wrap_t_scaled)

            
            
    
    plt.subplots(int(len(imWrap_s)/6)+1,6,figsize=(15,10))
    for ii in range(0,len(imWrap_t_wrap_t_s)):
        plt.subplot(int(len(imWrap_s)/6)+1,6,ii+1)
        plt.imshow(imWrap_t_wrap_t_scaled_s[ii])
    plt.show()
    plt.savefig('test.png')
