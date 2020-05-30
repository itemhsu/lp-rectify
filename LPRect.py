#depends on
#pip install  Pillow
#pip install  sklearn
#pip install  opencv-python
#pip install  matplotlib

import pandas as pd
import os, sys
import numpy as np
import math
from PIL import Image
import PIL
from sklearn import datasets,linear_model
import cv2


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def axbFit(x_train,y_train):
    regr=linear_model.LinearRegression()
    regr.fit(x_train.reshape(-1, 1),y_train.reshape(-1, 1))  
    r_squared = regr.score(x_train.reshape(-1, 1),y_train.reshape(-1, 1))
    return regr.coef_, r_squared

def imgWrapA(orgImg,a):
    column=orgImg.shape[1]
    row=orgImg.shape[0]
    pts1 = np.float32([[column/2,row/2],[column/2,row/4],[column/4,row/2-column/4*a]])
    pts2 = np.float32([[column/2,row/2],[column/2,row/4],[column/4,row/2]])
    M = cv2.getAffineTransform(pts1,pts2)
    imgWarpAffine = cv2.warpAffine(orgImg,M,(column,row))
    return imgWarpAffine


def v3GetTrelis(img, row, column, side, orgImg):
    inc=np.array(range(0-2,row-2))#2 = side
    reward= np.zeros((row,column),dtype=float)
    prev= np.zeros((row,column),dtype=float)
    orgImgM= np.ones((row,column,side*2+1),dtype=float)*(-200)
    columnTrace=column
    #columnTrace=4
    
    rewardM= np.ones((row,side*2+1),dtype=float)*(-1000)
    colInx=0
    orgImgM[1:  ,:,1]=orgImg[ :-1,:]+40.0
    orgImgM[ :  ,:,2]=orgImg[ :  ,:]
    orgImgM[ :-1,:,3]=orgImg[1:  ,:]+40.0
    #for i in range(0, 2*side+1):
    for i in range(1, columnTrace):  # for each stage
        colInx=i-1
        rewardM[1:  ,1]=reward[ :-1,colInx]
        rewardM[ :  ,2]=reward[ :  ,colInx]
        rewardM[ :-1,3]=reward[1:  ,colInx]
        vNewReward=rewardM-abs(orgImgM[:,colInx,:]-orgImg[:,i, None])
        vRelativeMaxIndex=np.argmax(vNewReward,axis=1)
        reward[:,i]=np.max(vNewReward,axis=1)+img[:,i]
        prev[:,i]=inc+vRelativeMaxIndex

    return reward, prev


def dpDdecode(img, row, column, side, orgImg):
    #reward, prev=vvGetTrelis(img, row, column, side, orgImg)
    reward, prev=v3GetTrelis(img, row, column, side, orgImg)
    #reward, prev=vGetTrelis(img, row, column, side, orgImg)
    #reward, prev=getTrelis(img, row, column, side, orgImg)
    columnTrace=column
    
    # traceback
    #rewardMax=np.max(reward[:,columnTrace-1])
    #rewardThread=rewardMax/2
    finalMaxIndex=np.argmax(reward[:,columnTrace-1])
    tmpIndex=finalMaxIndex
    data_r=np.zeros(columnTrace)
    data_r[columnTrace-1]=row-tmpIndex
        
    for i in range(columnTrace-1, 0, -1):  # for each stage
        #print(tmpIndex)
        #print(reward[tmpIndex,i])
        tmpIndex=int(prev[tmpIndex,i])
        #data_r[i-1]=row-tmpIndex
        data_r[i-1]=tmpIndex
    a, r_squared=axbFit(np.arange(len(data_r)-1),data_r[:-1])

    r_x_squared_min=9999.0
    r_x_squared_min_index=0
    data_r_min=np.zeros(columnTrace)
    r_x_squared_max=-9999.0
    r_x_squared_max_index=0
    data_r_max=np.zeros(columnTrace)
    a_x_max=0
    
    center=int(column/2)
    e=int(4*center/3)
    b=int(2*center/3)

    #find up line
    for ii in range(2,row-2):
        w=abs(ii/row-0.5)
        tmpIndex=ii
        data_r_x_up=np.zeros(columnTrace)
        data_r_x_up[columnTrace-1]=tmpIndex
        for jj in range(columnTrace-1, 0, -1):  # for each stage
            tmpIndex=int(prev[tmpIndex,jj])
            data_r_x_up[jj-1]=tmpIndex        
        if data_r_x_up[center]>3:
            centerUp=data_r_x_up[center]
            a_x_up, r_x_squared_up=axbFit(np.arange(e-b),data_r_x_up[b:e])
            break

    maxReward=-9999
    for ii in range(2,int(row/2)):
        tmpIndex=ii
        data_r_x_up_2=np.zeros(columnTrace)
        data_r_x_up_2[columnTrace-1]=tmpIndex
        tmpIndex2=tmpIndex
        for jj in range(columnTrace-1, 0, -1):  # for each stage
            tmpIndex2=int(prev[tmpIndex2,jj])
            data_r_x_up_2[jj-1]=tmpIndex2        
        if reward[ii,columnTrace-1]>maxReward and centerUp==data_r_x_up_2[center]:
            maxReward=reward[ii,columnTrace-1]
            data_r_x_up=data_r_x_up_2
            a_x_up, r_x_squared_up=axbFit(np.arange(e-b),data_r_x_up[b:e])
            
    
    #for finding down line
    for ii in range(row-1,0,-1):
        w=abs(ii/row-0.5)
        tmpIndex=ii
        data_r_x_down=np.zeros(columnTrace)
        data_r_x_down[columnTrace-1]=tmpIndex
        for jj in range(columnTrace-1, 0, -1):  # for each stage
            tmpIndex=int(prev[tmpIndex,jj])
            data_r_x_down[jj-1]=tmpIndex        
        if data_r_x_down[center]<row-3:
            a_x_down, r_x_squared_down=axbFit(np.arange(e-b),data_r_x_down[b:e])
            break
    data_r_x_down_center=data_r_x_down[int(len(data_r_x_down)/2)]
    data_r_x_up_center=data_r_x_up[int(len(data_r_x_up)/2)]
    
    if r_x_squared_down>0.98:
        a=a_x_down
        data_r_max=data_r_x_down
    elif r_x_squared_up>0.98:
        a=a_x_up
        data_r_max=data_r_x_up
    elif abs(data_r_x_down_center+data_r_x_up_center)>row:
        a=a_x_down
        data_r_max=data_r_x_down
    else:
        a=a_x_up
        data_r_max=data_r_x_up
    return data_r_max, a

def rectScale(im):
    myWidth = im.shape[0]
    myHeight = int(myWidth/2)
    nim = cv2.resize(im, (myWidth*2, myHeight*2), interpolation=cv2.INTER_AREA)
    return nim

def lp_parser(path):
    im = Image.open(path)
    pix_color = np.array(im)
    pix_gray = rgb2gray(pix_color)

    left=0
    right=im.size[0]
    top=0+3
    down=im.size[1]-3
    
    cropGrayLR=np.abs(pix_gray[0:-1,left:right]-pix_gray[1:,left:right])
    waveLR, a=dpDdecode(cropGrayLR,cropGrayLR.shape[0],cropGrayLR.shape[1],2, pix_gray[1:,left:right])
    imWrap=imgWrapA(pix_color,a)
    gray = cv2.cvtColor(imWrap,cv2.COLOR_BGR2GRAY)
    y,gray_mask = cv2.threshold(gray,1,1,cv2.THRESH_BINARY)
    
    imWrap_t = imWrap.copy().transpose(1,0,2) 
    gray_mask_t  = gray_mask.copy().transpose(1,0) 
    pix_gray_t=rgb2gray(imWrap_t)
    cropGrayTD=np.abs(pix_gray_t[0:-1,top:down]-pix_gray_t[1:,top:down])
    cropGrayTD_masked=np.multiply(cropGrayTD, gray_mask_t[0:-1,top:down]) 
    cropGrayTD_masked2=np.multiply(cropGrayTD_masked, gray_mask_t[0:-1,top-1:down-1]) 
    cropGrayTD_masked3=np.multiply(cropGrayTD_masked2, gray_mask_t[0:-1,top-2:down-2]) 
    cropGrayTD_masked4=np.multiply(cropGrayTD_masked3, gray_mask_t[0:-1,top+1:down+1]) 
    cropGrayTD_masked5=np.multiply(cropGrayTD_masked4, gray_mask_t[0:-1,top+2:down+2]) 
    waveLR, a_t=dpDdecode(cropGrayTD_masked5,cropGrayTD_masked5.shape[0],cropGrayTD_masked5.shape[1],2, pix_gray_t[1:,top:down])
    imWrap_t_wrap=imgWrapA(imWrap_t,a_t)
    imWrap_t_wrap_t = imWrap_t_wrap.copy().transpose(1,0,2)   
    imWrap_t_wrap_t_scaled=rectScale(imWrap_t_wrap_t)
    
    return waveLR,pix_gray[0:-1,left:right],a, imWrap, imWrap_t_wrap_t, imWrap_t_wrap_t_scaled


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    lpRect_s=[]
    pix_gray_s=[]
    imWrap_s=[]
    imWrap_t_wrap_t_s=[]
    imWrap_t_wrap_t_scaled_s=[]
    for root,unkown,fNames in os.walk('img'):
        for f in fNames:
            strFpath="img/{}".format(f)
            lpRect, pix_gray, a, imWrap, imWrap_t_wrap_t, imWrap_t_wrap_t_scaled=lp_parser(strFpath)
            lpRect_s.append(lpRect)
            pix_gray_s.append(pix_gray)
            imWrap_s.append(imWrap)
            imWrap_t_wrap_t_s.append(imWrap_t_wrap_t)
            imWrap_t_wrap_t_scaled_s.append(imWrap_t_wrap_t_scaled)
    
    
    plt.subplots(int(len(imWrap_s)/6)+1,6,figsize=(15,10))
    for ii in range(0,len(imWrap_t_wrap_t_s)):
        plt.subplot(int(len(imWrap_s)/6)+1,6,ii+1)
        plt.imshow(imWrap_t_wrap_t_scaled_s[ii])
    plt.show()
    plt.savefig('test.png')
