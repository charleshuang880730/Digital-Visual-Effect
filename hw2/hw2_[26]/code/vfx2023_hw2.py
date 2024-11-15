# -*- coding: utf-8 -*-
"""VFX2023_HW2.ipynb

# Import package
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import scipy
from sklearn.preprocessing import normalize
import math
import os

"""#Cylincal Warping"""

def Bilinear(img,back_x,back_y):  # Bilinear
    # img = np.pad(img, ((0,1), (0,1), (0,0)), 'edge')
    l = math.floor(back_x)
    k = math.floor(back_y)
    # print(l,k)
    if l >= img.shape[1]-1 or k>= img.shape[0]-1 or l <= 0 or k<=0 :
        return np.zeros((1,1,3))
    a = back_x - l
    b = back_y - k
    return (1-a)*(1-b)*img[k,l,:] + a*(1-b)*img[k,l+1,:] + (1-a)*b*img[k+1,l,:] + a*b*img[k+1,l+1,:]

def cylincal_warp(img,f):
    img_warp = np.zeros(img.shape)
    shift_y = img_warp.shape[0]//2
    shift_x = img_warp.shape[1]//2
    for y in range(-shift_y,shift_y):
        for x in range(-shift_x,shift_x):
            value  = Bilinear(img,math.tan(x/f)*f+shift_x, y/f*((x ** 2 + f **2) ** (1/2))+shift_y)
            # print(math.tan(x/f)*f)
            # print(y/f*((x ** 2 + f **2) ** (1/2)))
            # print(value)
            img_warp[y+shift_y,x+shift_x,:] = value
    return img_warp.astype('uint8')


"""# Feature detection"""

def non_maximum_suppression(harris, threshold):
    # 去除小於 thrshold 的點
    possible_corners = np.where(harris <= threshold, 0, 1)
    # 對每一個點去做判斷
    for i in range(1, harris.shape[0]-1):
        for j in range(1, harris.shape[1]-1):
            if possible_corners[i, j] == 1:
                # 判斷是否為 local maximum
                maximum = True
                for m in range(i-1, i+2):
                    for n in range(j-1, j+2):
                        if harris[m, n] > harris[i, j]:
                            maximum = False
                            break
                    if not maximum:
                        break

                # 不是 maximum 則標示為 0
                if not maximum:
                    possible_corners[i, j] = 0

    # 得到最後的 corner array
    corners = np.where(possible_corners == 1)
    # print("test corners:\n", corners)

    return corners

def harris_corner(img, kernel_size, k , sigma):

  counters = 0

  # 計算 gradient
  Iy, Ix = np.gradient(img)

  Ix2 = np.multiply(Ix, Ix)
  Iy2 = np.multiply(Iy, Iy)
  Ixy = np.multiply(Ix, Iy)

  # 對 matrix M 進行高斯加權
  S_XX = cv2.GaussianBlur(Ix2, (kernel_size, kernel_size), sigma)
  S_YY = cv2.GaussianBlur(Iy2, (kernel_size, kernel_size), sigma)
  S_XY = cv2.GaussianBlur(Ix * Iy, (kernel_size, kernel_size), sigma)

  # 計算 harris response
  harris = S_XX * S_YY - np.square(S_XY) - k * (np.square(S_XX + S_YY))
  
  # 計算 threshold 
  threshold = np.mean(harris)

  # 進行 non maximum suppression 
  corners = non_maximum_suppression(harris, threshold)

  # 畫出 feature point
  for pt in zip(*corners[::-1]):
    counters += 1
    cv2.circle(img, pt, 3, (0, 0, 255), -1)

  # print("testing image")
  print("Corners lens: ", counters)
  # plt.imshow(img)
  # plt.show()

  return zip(*corners[::-1])


"""# Feature description"""


def Find_major(x,y,all_hist_pad,magnitude,pad_size,GauKernel_size=11): #input all_hist output major orientation  #沒有實作有兩個major orientation
  # 找出major orientation
  xlocal = x + pad_size
  ylocal = y + pad_size
  patches = all_hist_pad[:,ylocal-pad_size:ylocal+pad_size+1,xlocal-pad_size:xlocal+pad_size+1]
  GauKernel=np.zeros((GauKernel_size,GauKernel_size))
  GauKernel[GauKernel_size//2,GauKernel_size//2]=1
  GauKernel = cv2.GaussianBlur(GauKernel,(GauKernel_size,GauKernel_size),0)
  # print(GauKernel)
  # print(patches.shape)
  angle = 0
  vote = 0
  for i in range(len(patches)):
    local_vote = np.sum(patches[i,:,:] * GauKernel)
    if local_vote > vote:
      vote = local_vote
      angle = i

  return angle

def make_description(x_rot,y_rot,all_hist_rotated):
  all_hist_num = all_hist_rotated.shape[0]
  bin8_index = [all_hist_num//8*i for i in range(0,9)]
  small_patch_index_shift = [-8,-4,0,4]
  local_des= []
  for yshift in small_patch_index_shift:
    for xshift in small_patch_index_shift:
      for i in range(len(bin8_index)):
        if i == 8:
          break
        value = np.sum(all_hist_rotated[bin8_index[i]:bin8_index[i+1],y_rot-yshift:y_rot-yshift+4,x_rot-xshift:x_rot-xshift+4])
        # print(all_hist_rotated[bin8_index[i]:bin8_index[i+1],y_rot-yshift:y_rot-yshift+4,x_rot-xshift:x_rot-xshift+4])
        # print(value)
        local_des.append(value)
  
  local_des = np.array(local_des)
  local_des = np.clip(((local_des)/ np.linalg.norm(local_des)),0,0.2)
  local_des /= np.linalg.norm(local_des)
  
  # print(local_des)
  return local_des

  # to let orientation hist 8 bins


def sift_descriptor(points,sobelx,sobely,bin_major = 36):
  magnitude = np.hypot(sobelx, sobely)
  theta = sobely / sobelx
  theta = np.arctan(theta)
  theta = np.rad2deg(theta)
  theta[sobelx<0]+=180  # 座標y正是往下 x正是往右 算角度是順時鐘加
  theta[theta<0]+=360
  # print(theta)
  Each_bin_angle = 360.0 / bin_major
  theta_hist = (theta) // int(Each_bin_angle)
  all_hist = np.zeros((bin_major,sobelx.shape[0],sobelx.shape[1])) #so x,y就跟圖片的相反了
  for i in range(bin_major):
    all_hist[i] = 1 * (theta_hist==i)
  
  
  despts = []
  
  pad_size = 11 // 2
  all_hist_pad = np.pad(all_hist,((0,0),(pad_size,pad_size),(pad_size,pad_size)),'constant',constant_values=0)
  # print(all_hist_pad.shape)
  for x, y in points:
    x = round(x)
    y = round(y)
    des = []
    angle = Find_major(x,y,all_hist_pad,magnitude,pad_size)
    
    # 順時鐘 要抵銷所以剛好逆時鐘 + 的 
    # 重新排列histogram，等效於旋轉原始圖片
    all_hist_rotated = np.concatenate((all_hist[angle:,:,:], all_hist[:angle,:,:]), axis=0)
    # print(all_hist_rotated.shape)
    # print(x, y, angle)
  
    
    # x_rot, y_rot = rotate_vector([x,y],angle)
    # print(all_hist_rotated)
    # 找出128維向量
    local_des = make_description(x,y,all_hist_rotated)
    despts.append(local_des)
    del all_hist_rotated
    # print(len(despts))
  print('finish making desription')
  return np.array(despts)

"""# Feature Matching

"""

def Matching(kp1, kp2, des1, des2, ratio=0.8):
  #input keypoints, desriptions
  #output matches [match1,match2,...] each match has [pt_img1,pt_img2]
  matches = []
  for i in range(des1.shape[0]):
    dist = []
    for j in range(des2.shape[0]):
      dist.append([np.linalg.norm(des1[i]-des2[j],1),kp1[i],kp2[j]])
    dist.sort(key = lambda x: x[0])
    # print(dist)
    if dist[0][0] < dist[1][0] * ratio:
      # print(dist[0][1],dist[0][2])
      matches.append([dist[0][1],dist[0][2]])
  
  return matches
    

'''
Homography
'''
# def homography(pts1, pts2):

#     # 兩個特徵點的 list shape 必須相同
#     assert pts1.shape[0] == pts2.shape[0], "points shape should be equal"
    
#     n = pts1.shape[0]

#     # 矩陣 A 是 2n * 4 的，求解 least squares solution
#     A = np.zeros((2 * n, 4))
#     B = np.zeros((2 * n, 1))
    
#     for i in range(n):
#         x, y = pts1[i, 0], pts1[i, 1]
#         u, v = pts2[i, 0], pts2[i, 1]
        
#         A[2*i, :] = [x, 0, 1, 0]
#         B[2*i, :] = [u]
        
#         A[2*i+1, :] = [0, y, 0, 1]
#         B[2*i+1, :] = [v]

#     # 求解 Ax=B 的 least squares solution
#     scaling_translation = np.linalg.lstsq(A, B, rcond=None)[0]
    
#     sx = scaling_translation[0][0]
#     sy = scaling_translation[1][0]
#     tx = scaling_translation[2][0]
#     ty = scaling_translation[3][0]

#     # 製作 scaling 與 translation 的 3x3 矩陣
#     H = np.array([[sx, 0, tx], 
#                   [0, sy, ty],
#                   [0, 0, 1]])
#     return H

def homography(pts1, pts2, mode='ST'):

    # 兩個特徵點的 list shape 必須相同
    assert pts1.shape[0] == pts2.shape[0], "points shape should be equal"
    if mode == 'H':
      n = pts1.shape[0]

      # A 是一個 2n * 9 的矩陣，求解 Ah = b 的 least square solution
      A = np.zeros((2 * n, 9))
      for i in range(n):
          x, y = pts1[i, 0], pts1[i, 1]
          u, v = pts2[i, 0], pts2[i, 1]
          A[2*i, :] = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
          A[2*i+1, :] = [0, 0, 0, x, y, 1, -v*x, -v*y, -v]

      _, _, V = np.linalg.svd(A)
      temp = V[-1, :]
      H = temp.reshape((3, 3))

      return H / H[2, 2]

    else:
      n = pts1.shape[0]

      # 矩陣 A 是 2n * 4 的，求解 least squares solution
      A = np.zeros((2 * n, 4))
      B = np.zeros((2 * n, 1))
      
      for i in range(n):
          x, y = pts1[i, 0], pts1[i, 1]
          u, v = pts2[i, 0], pts2[i, 1]
          
          A[2*i, :] = [x, 0, 1, 0]
          B[2*i, :] = [u]
          
          A[2*i+1, :] = [0, y, 0, 1]
          B[2*i+1, :] = [v]

      # 求解 Ax=B 的 least squares solution
      scaling_translation = np.linalg.lstsq(A, B, rcond=None)[0]
      
      sx = scaling_translation[0][0]
      sy = scaling_translation[1][0]
      tx = scaling_translation[2][0]
      ty = scaling_translation[3][0]

      # 製作 scaling 與 translation 的 3x3 矩陣
      H = np.array([[sx, 0, tx], 
                    [0, sy, ty],
                    [0, 0, 1]])
    return H

def ransac(pts1, pts2, mode, max_iter=10000, threshold=5):
  
    assert pts1.shape[0] == pts2.shape[0], "points shape should be equal."

    n = pts1.shape[0]
    best_H = None
    best_inliers = 0
    best_mask = np.zeros(n, dtype=bool)
    # threshold = np.inf

    for i in range(max_iter):
        # 隨機選 4 個點
        idx = np.random.choice(n, 4, replace=False)
        # 計算 homography matrix
        H = homography(pts1[idx], pts2[idx],mode)
        """print(H)
        print()"""

        # 計算點和 transform 後的點的距離
        pts1_h = np.concatenate([pts1, np.ones((n, 1))], axis=1) @ H.T
        pts1_h = pts1_h[:, :2] / pts1_h[:, 2:]
        dist = np.linalg.norm(pts1_h - pts2, axis=1)

        # 如果有找到更好的 homography matrix 就做 update
        mask_i = dist < threshold
        if np.sum(mask_i) > best_inliers:
            print("RRRRRR")
            best_H = homography(pts1[mask_i], pts2[mask_i],mode)
            best_mask = mask_i
            best_inliers = np.sum(mask_i)
            # threshold = dist

    return best_H, best_mask

"""Warping"""

def warping(src, dst, H, xmax, ymax):
    
    # find the coordinates of the non-zero elements in the image
    # coords = np.nonzero(np.any(src != [0, 0, 0], axis=-1))

    # # get the minimum and maximum x and y coordinates of the non-zero elements
    # a, b = np.min(coords, axis=1)
    # c, d = np.max(coords, axis=1)

    # crop the image to the non-zero region
    # src = src[a+10:c-10, b+10:d-10]
    # plt.imshow(src)
    # plt.show()
    
    h_src, w_src, _ = src.shape
    h_dst, w_dst, _ = dst.shape
    H_inv = np.linalg.inv(H)

    # meshgrid the (x, y) coordinate pairs
    x, y = np.meshgrid(np.arange(0, xmax), np.arange(-ymax//4, ymax//4*3))
    x, y = x.flatten().astype(int), y.flatten().astype(int)
    # print(x)
    # print(y)
    # reshape the destination pixels as N x 3 homogeneous coordinate
    v = np.stack([x, y, np.ones_like(x)])
    u = np.matmul(H_inv, v)
    u /= u[2]

    mask = np.logical_or(np.logical_or(u[0] < 0, u[0] > w_src - 1), np.logical_or(u[1] < 0, u[1] > h_src - 1))
    v_x, v_y, u_x, u_y = np.delete(v[0], mask), np.delete(v[1], mask), np.delete(u[0], mask), np.delete(u[1], mask)

    # loop through the valid pixels and compute the interpolated value for each pixel
    for i in range(v_x.shape[0]):
        x, y = v_x[i], v_y[i]+ymax//4
        back_x, back_y = u_x[i], u_y[i]
        # print(dst[y,x])
        if not (dst[y,x] == np.zeros((1,3))).all():
          # print('dst[y,x]',dst[y,x])
          # print('Bilinear(src, back_x, back_y)',Bilinear(src, back_x, back_y))
          dst[y,x] = np.maximum(dst[y,x], Bilinear(src, back_x, back_y))
        else:
          dst[y, x] = Bilinear(src, back_x, back_y)

    return dst

'''
To cylinical coordinates
'''
DATA_NAME = 'Roof'  

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
path_cylinical = os.path.join('..','data',f'{DATA_NAME}_warp')
path_data = os.path.join('..','data')
if not os.path.exists(path_cylinical):
  os.mkdir(path_cylinical)
dir = os.path.join('..','data',DATA_NAME,'*png') #parrington:jpg, Roof png
files = sorted(glob.glob(dir))
for i, file in enumerate(files):
    img = cv2.imread(file)
    print(img.shape)
    img_warp = cylincal_warp(img,652)  #Roof:652  #parrington:702
    if i < 10:
      cv2.imwrite(os.path.join(path_cylinical,f'{DATA_NAME}0{i}.png'),img_warp)
    else:
      cv2.imwrite(os.path.join(path_cylinical,f'{DATA_NAME}{i}.png'),img_warp)
      
"""# Stitch all image"""
files = glob.glob(os.path.join(path_cylinical,'*png'))
files= sorted(files)
# files=files[1:]
print(files)

base_color = cv2.imread(files[-1])
imgs = [cv2.imread(path) for path in files]
h_max = max([im.shape[0] for im in imgs]) * 2
w_max = sum([im.shape[1] for im in imgs])

# create the final stitched canvas
destination = np.zeros((h_max, w_max, base_color.shape[2]), dtype=np.uint8)

# find the coordinates of the non-zero elements in the image
coords = np.nonzero(np.any(base_color != [0, 0, 0], axis=-1))
# get the minimum and maximum x and y coordinates of the non-zero elements
y_min, x_min = np.min(coords, axis=1)
y_max, x_max = np.max(coords, axis=1)
# crop the image to the non-zero region
base_color = base_color#[y_min+10:y_max-10, x_min+10:x_max-10]

destination[h_max//4:base_color.shape[0]+h_max//4, :base_color.shape[1]] = base_color
best_H_total = np.eye(3)
start_position = 0

for i in range(len(files)-1, 1, -1):
  base_color = cv2.imread(files[i])
  new_color = cv2.imread(files[i-1])
  base = cv2.cvtColor(base_color,cv2.COLOR_BGR2GRAY)
  new = cv2.cvtColor(new_color,cv2.COLOR_BGR2GRAY)
  # img1 gradient
  sobelx_base = cv2.Sobel(base,cv2.CV_64F,1,0,ksize=5)
  sobely_base = cv2.Sobel(base,cv2.CV_64F,0,1,ksize=5)

  # prevent 0 value
  sobelx_base[sobelx_base==0] = 1
  sobely_base[sobely_base==0] = 1

  # img2 gradient
  sobelx_new = cv2.Sobel(new,cv2.CV_64F,1,0,ksize=5)
  sobely_new = cv2.Sobel(new,cv2.CV_64F,0,1,ksize=5)

  # prevent 0 value
  sobelx_new[sobelx_new==0] = 1
  sobely_new[sobely_new==0] = 1

  base_coordinates = harris_corner(base, kernel_size=9, k=0.04, sigma=3)
  new_coordinates = harris_corner(new, kernel_size=9, k=0.04, sigma=3)

  feature_list_base = []
  feature_list_new = []
  for pt in base_coordinates:
    feature_list_base.append(list(pt))
  for pt in new_coordinates:
    feature_list_new.append(list(pt))

  des_base = sift_descriptor(feature_list_base, sobelx_base, sobely_base, 72)
  des_new = sift_descriptor(feature_list_new,sobelx_new,sobely_new, 72)
  matches = Matching(feature_list_new, feature_list_base, des_new, des_base,ratio=0.75)
  print(matches)

  match_list_base = []
  match_list_new = []
  for mpts in matches:
    match_list_base.append(mpts[1])
    match_list_new.append(mpts[0])

  feature_array_base = np.array(match_list_base)
  feature_array_new = np.array(match_list_new)

  best_H, inliers_mask = ransac(feature_array_new, feature_array_base,'H',threshold=5) #Roof: H, parrington: ST
  best_H_total = best_H_total.dot(best_H)

  destination_new = warping(new_color, destination, best_H_total, w_max, h_max)
  cv2.imwrite(os.path.join(path_data,f'panorama_{DATA_NAME}.png'), destination_new)
  destination = destination_new

  start_position += base_color.shape[1]
