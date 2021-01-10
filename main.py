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


width = 1295*2
height = 746*2
imgfold = './dataset/test/sample_good'
grodtruth_fold = './dataset/test/sample_good_Groundtruth'

# 建立資料夾
if not os.path.isdir('./result'):
    os.makedirs('./result')

time_cost_list = []
IOU_list = []
IOU_sum = 0
time_sum = 0

for img in tqdm(range(7)):
    time_start = time.time()
    input_before = cv2.imread(os.path.join(imgfold, (str(img+1) + '_before.jpg')))
    input_after = cv2.imread(os.path.join(imgfold, (str(img+1) + '_after.jpg')))
    answer = cv2.imread(os.path.join(grodtruth_fold, (str(img+1) + '_correct.jpg')))

    input_before = cv2.cvtColor(input_before, cv2.COLOR_BGR2GRAY)   #灰階
    input_after = cv2.cvtColor(input_after, cv2.COLOR_BGR2GRAY)   #灰階
    answer = cv2.cvtColor(answer, cv2.COLOR_BGR2GRAY)   #灰階

    input_before = cv2.resize(input_before, (width, height), interpolation=cv2.INTER_AREA)   #圖片縮放
    input_after = cv2.resize(input_after, (width, height), interpolation=cv2.INTER_AREA)   #圖片縮放
    answer = cv2.resize(answer, (width, height), interpolation=cv2.INTER_AREA)   #圖片縮放

    result = cv2.subtract(input_after, input_before)#圖片相減

    ret, result = cv2.threshold(result, 0,255,cv2.THRESH_OTSU)  #二值化
    time_end = time.time()
    time_cost = time_end - time_start
    time_sum += time_cost
    time_cost_list.append(time_cost)

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
    IOU_sum += IOU
    IOU_list.append('%.2f'%IOU)
    # print(IOU)
    # cv2.waitKey(0)

# average
IOU_list.append('%.2f'%(IOU_sum/7))
time_cost_list.append(time_sum/7)

df = pd.DataFrame({'IOU':IOU_list, 'Time':time_cost_list})
print(df)