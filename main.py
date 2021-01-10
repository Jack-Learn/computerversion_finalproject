import cv2
import os

def count_area(img, width, height):
    area = 0
    for i in range(height):
        for j in range(width):
            if img[i, j] == 255:
                area += 1
    return area


width = 1295*2
height = 746*2
imgfold = r'D:\Doucuments\computer_version\Final_Project\Dataset\test\sample_good'
grodtruth_fold = r'D:\Doucuments\computer_version\Final_Project\Dataset\test\sample_good_Groundtruth'
IOU_list = []
IOU_sum = 0
for img in range(7):
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
    # cv2.imshow(('result_'+str(img+1)), result)

    #IOU
    bitwiseAnd  = cv2.bitwise_and(result, answer)    #AND
    area_and = count_area(bitwiseAnd, width, height)
    bitwise_OR  = cv2.bitwise_or(result, answer)    #OR
    area_or = count_area(bitwise_OR, width, height)
    IOU = (area_and / area_or)*100
    IOU_sum += IOU
    IOU_list.append(IOU)
    print(IOU)
    # cv2.waitKey(0)
print(IOU_list)
print(IOU_sum/7)
