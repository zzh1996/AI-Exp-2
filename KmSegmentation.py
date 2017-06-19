#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import Image

def kmeans(V,k):
    # choose k random centers from V
    nums=np.arange(len(V))
    np.random.shuffle(nums)
    centers=V[nums[:k]]
    last_centers=None
    while last_centers is None or not np.allclose(centers,last_centers):
        # find closest cluster for each v
        clusters=np.argmin(((V-centers[:,np.newaxis,:])**2).sum(2),0)
        last_centers=centers[:]
        # calculate new centers
        centers=np.array([V[clusters==i].mean(0) for i in range(k)])
        print centers
    return centers,clusters


img=Image.open('Sea.jpg')
data=np.array(img).reshape(-1,3)

for k in [2,4,8,16]:
    centers,clusters=kmeans(data,k)
    new_data=centers[clusters].reshape(img.height,img.width,3)
    print new_data
    new_img=Image.fromarray(np.asarray(np.clip(new_data,0,255),dtype='uint8'))
    new_img.show()

