import cv2
import os
import pandas as pd
import time
from tqdm import tqdm

def count_area(img, width, height):
    area = 0
    for i in range(height):
        for j in range(width):
            if img[i, j] == 255:
                area += 1
    return area




width = int(2590/2)
height = int(1942/2)
imgfold = './dataset/test/sample_good'
grodtruth_fold = './dataset/test/sample_good_Groundtruth'

# 建立資料夾
if not os.path.isdir('./result'):
    os.makedirs('./result')

time_cost_list = []
IOU_list = []
IOU_sum = 0
time_sum = 0
index_list = ['1', '2', '3', '4', '5', '6', '7', 'average']
for img in tqdm(range(7)):
    time_start = time.time()
    input_before = cv2.imread(os.path.join(imgfold, (str(img+1) + '_before.jpg')))
    input_after = cv2.imread(os.path.join(imgfold, (str(img+1) + '_after.jpg')))
    answer = cv2.imread(os.path.join(grodtruth_fold, (str(img+1) + '_correct.jpg')))

    input_before = cv2.resize(input_before, (width, height), interpolation=cv2.INTER_AREA)   #圖片縮放
    input_after = cv2.resize(input_after, (width, height), interpolation=cv2.INTER_AREA)   #圖片縮放
    answer = cv2.resize(answer, (width, height), interpolation=cv2.INTER_AREA)   #圖片縮放

    input_before = cv2.cvtColor(input_before, cv2.COLOR_BGR2GRAY)   #灰階
    input_after = cv2.cvtColor(input_after, cv2.COLOR_BGR2GRAY)   #灰階
    answer = cv2.cvtColor(answer, cv2.COLOR_BGR2GRAY)   #灰階
    sub = cv2.subtract(input_after, input_before)#圖片相減
    # cv2.imshow('sub', sub)
    # 高思濾波
    img_Guassian = cv2.GaussianBlur(sub,(7,7),0)
    img_Guassian = cv2.GaussianBlur(img_Guassian,(7,7),0)
    img_Guassian = cv2.GaussianBlur(img_Guassian,(5,5),0)
    img_Guassian = cv2.GaussianBlur(img_Guassian,(5,5),0)
    img_Guassian = cv2.GaussianBlur(img_Guassian,(3,3),0)
    img_Guassian = cv2.GaussianBlur(img_Guassian,(3,3),0)
    # cv2.imshow('img_Guassian', img_Guassian)
    # cv2.waitKey(0)
    _, result = cv2.threshold(img_Guassian, 0,255,cv2.THRESH_OTSU)  #二值化
    
    


    # 顯示結果與存檔
    # cv2.imshow(('result_'+str(img+1)), result)
    saveimg = os.path.join('./result', 'result_{}.jpg'.format(str(img+1)))
    cv2.imencode('.jpg', result)[1].tofile(saveimg)

    #IOU
    bitwiseAnd  = cv2.bitwise_and(result, answer)    #AND
    area_and = count_area(bitwiseAnd, width, height)
    bitwise_OR  = cv2.bitwise_or(result, answer)    #OR
    area_or = count_area(bitwise_OR, width, height)
    IOU = (area_and / area_or)*100
    # print('\n',IOU)
    IOU_sum += IOU
    IOU_list.append('%.2f'%IOU + '%')
    # print(IOU)
    # cv2.waitKey(0)
    time_end = time.time()
    time_cost = time_end - time_start
    time_sum += time_cost
    time_cost_list.append(time_cost)

# average
IOU_list.append('%.2f'%(IOU_sum/6)+'%')
time_cost_list.append(time_sum/6)

df = pd.DataFrame({'IoU(%)':IOU_list, 'Time(s)':time_cost_list}, index= index_list)
print(df)