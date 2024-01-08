from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import subprocess

from gradio_client import Client

import cv2
import time

#스타일 바꾸기 전 전체 신발 이미지
shoes = input('style 바꿀 신발 선택하세요(cro,newbal,nike,adidas,mlb 중 택1): ')
original = cv2.imread(f'shoes/{shoes}.jpg')

#style 바꿀 mask 이미지
if shoes == "nike":
    mask_choice = input('style 바꿀 mask 부분 선택하세요(base,lace,line,nike,sole): ')
    filename1 = f'mask/segmentation_{shoes}_{mask_choice}.png'
    mask = cv2.imread(filename1,cv2.IMREAD_GRAYSCALE)
elif shoes == "newbal":
    mask_choice = input('style 바꿀 mask 부분 선택하세요(back,base,fb,lace,N,sole 등..): ')
    filename1 = f'mask/segmentation_{shoes}_{mask_choice}.png'
    mask = cv2.imread(filename1,cv2.IMREAD_GRAYSCALE)
elif shoes == "adidas":
    mask_choice = input('style 바꿀 mask 부분 선택하세요(base,lace,logo,sole,top,top_point 등..): ')
    filename1 = f'mask/segmentation_{shoes}_{mask_choice}.png'
    mask = cv2.imread(filename1,cv2.IMREAD_GRAYSCALE)
elif shoes == "mlb":
    mask_choice = input('style 바꿀 mask 부분 선택하세요(base_back,base_end,base_front,front,lace_down,logo,sole,sole_fb,top 등..): ')
    filename1 = f'mask/segmentation_{shoes}_{mask_choice}.png'
    mask = cv2.imread(filename1,cv2.IMREAD_GRAYSCALE)
else:
    mask_choice = input('style 바꿀 mask 부분 선택하세요(base,sole,top 등..): ')
    filename1 = f'mask/segmentation_{shoes}_{mask_choice}.png'
    mask = cv2.imread(filename1,cv2.IMREAD_GRAYSCALE)

#바꾸고자 하는 style 이미지
style1 = input('원하는 스타일을 선택하세요(aurora,galaxy_1,2,3,4,5,6): ')
filename2 = f'style/{style1}.jpg'

start = time.time()

command = f"python CAP-VSTNet/image_transfer.py --mode photorealistic --ckpoint CAP-VSTNet/checkpoints/photo_image.pt --content shoes/{shoes}.jpg --style {filename2}"

subprocess.run(command, shell=True)

# Composition

output_mask_style = cv2.imread(f'output/{shoes}_{style1}.png')

transfer = cv2.copyTo(output_mask_style,mask,original) #3개 이미지 곱 style, mask, original

NAME = f'results/output_mask_{shoes}_{style1}'

cv2.imwrite(f'{NAME}.png',transfer)
print("Style transfer success")

# 3D preprocessing

command = f"python dreamgaussian/process.py {NAME}.png"

subprocess.run(command, shell=True)
print("3D Preprocessing success")

# 3D multi view

client = Client("https://one-2-3-45-one-2-3-45.hf.space/")

input_img_path = f'{NAME}_rgba.png'

elevation_angle_deg = client.predict(
	input_img_path,
	True,		# image preprocessing
	api_name="/estimate_elevation"
)

print("3D mulriview success")

Elevation = elevation_angle_deg


# 3D reconstruction

command = f"python dreamgaussian/main.py --config dreamgaussian/configs/image.yaml input={input_img_path} save_path={NAME} elevation={Elevation} force_cuda_rast=True"
subprocess.run(command, shell=True)
print("3D reconstruction stage1 success")

command = f"python dreamgaussian/main2.py --config dreamgaussian/configs/image.yaml input={input_img_path} save_path={NAME} elevation={Elevation} force_cuda_rast=True"
subprocess.run(command, shell=True)
print("3D reconstruction stage2 success")

# 3D obj

command = f"python -m kiui.render logs/{NAME}.obj --save_video {NAME}.mp4 --wogui --force_cuda_rast"
subprocess.run(command, shell=True)
print("3D onj generation success")
print("time :", time.time() - start)












