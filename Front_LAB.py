from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import subprocess

from gradio_client import Client

import cv2
import time



from itertools import combinations
from PIL import Image
import os

def combine_and_save_images(images, num_combinations, output_path, shoes):
    # 모든 조합을 생성하여 이미지를 조합하고 저장합니다.
    for o, combination in enumerate(combinations(images, num_combinations)):
        combined_image = Image.new("L", images[0].size, 0)  # 첫 번째 이미지와 동일한 크기의 흰 배경 이미지를 생성합니다.

        for image in combination:
            combined_image = np.array(combined_image)
            image = np.array(image)
            result_array = combined_image + image
            result_array = np.where(result_array >= 10, 255, 0)
            combined_image = Image.fromarray(result_array.astype('uint8'))

        # 조합된 이미지를 저장합니다.
        output_filename = f"combined_{num_combinations}_{o}_{shoes}_images.png"
        output_filepath = os.path.join(output_path, output_filename)
        combined_image.save(output_filepath)

def combine_images(input_path, output_path,shoes):
    # input_path에 있는 모든 이미지 파일 목록을 가져옵니다.
    image_files = [f for f in os.listdir(input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = [x for x in image_files if shoes in x]
    # 이미지 파일이 없으면 종료합니다.
    if not image_files:
        print("No image files found in the input path.")
        return

    # 이미지들을 저장할 리스트를 초기화합니다.
    images = []

    # 각 이미지를 흑백으로 변환하여 리스트에 추가합니다.
    for image_file in image_files:
        image_path = os.path.join(input_path, image_file)
        image = Image.open(image_path).convert("L")
        images.append(image)

    # 이미지를 2개, 3개, 4개의 조합으로 더하여 저장합니다.
    for num_combinations in [1, 2, 3, 4]:
        combine_and_save_images(images, num_combinations, output_path, shoes)

input_path = "mask"
output_path = "mask_combi"
os.makedirs(output_path, exist_ok=True)
shoes_lst = os.listdir("shoes")
# 이미지를 조합하고 저장하는 함수 호출
for shoes in shoes_lst:
    combine_images(input_path, output_path,shoes.split(".")[0])



#스타일 바꾸기 전 전체 신발 이미지
# shoes = input('style 바꿀 신발 선택하세요(cro,newbal,nike,adidas,mlb 중 택1): ')
# original = cv2.imread(f'shoes/{shoes}.jpg')
import os
for shoes in os.listdir("shoes"):
    original = cv2.imread(os.path.join("shoes", shoes))
    shoe=os.path.join("shoes", shoes)
    shoes = shoes.split(".")[0]
    print(shoes)
    mask_lst = os.listdir("mask_combi")
    # print(mask_lst,"d")
    mask_lst = [x for x in mask_lst if shoes in x]
    # print(mask_lst,"d")
    for num_mask, mask_choice in enumerate(mask_lst):
        original = cv2.imread(os.path.join("shoes", shoes+".jpg"))
        print(mask_choice)
        filename1 = f'mask_combi/{mask_choice}'
        style_lst = os.listdir("style")
        for style1 in style_lst:
            print(style1)
            if style1==""or "ipynb" in style1:
                continue;
            filename2 = f'style/{style1}'

            start = time.time()

            command = f"python CAP-VSTNet/image_transfer.py --mode photorealistic --ckpoint CAP-VSTNet/checkpoints/photo_image.pt --content {shoe} --style {filename2}"

            subprocess.run(command, shell=True)

            # Composition
            mask_choice2 = mask_choice.split("_")[-1].split(".")[0]
            style3 = style1.split(".")[0]
            output_mask_style = cv2.imread(f'output/{shoes}_{style3}.png')
            mask = cv2.imread(filename1,cv2.IMREAD_GRAYSCALE)
            transfer = cv2.copyTo(output_mask_style,mask,original) #3개 이미지 곱 style, mask, original

            NAME = f'results/output_mask_{shoes}_{style3}_{mask_choice2}_{num_mask}'

            cv2.imwrite(f'{NAME}.png',transfer)
            print("Style transfer success")
        del mask

            # 3D preprocessing

            # command = f"python dreamgaussian/process.py {NAME}.png"

            # subprocess.run(command, shell=True)
            # print("3D Preprocessing success")

            # # 3D multi view

            # client = Client("https://one-2-3-45-one-2-3-45.hf.space/")

            # input_img_path = f'{NAME}_rgba.png'

            # elevation_angle_deg = client.predict(
            #     input_img_path,
            #     True,      # image preprocessing
            #     api_name="/estimate_elevation"
            # )

            # print("3D mulriview success")

            # Elevation = elevation_angle_deg


            # # 3D reconstruction

            # command = f"python dreamgaussian/main.py --config dreamgaussian/configs/image.yaml input={input_img_path} save_path={NAME} elevation={Elevation} force_cuda_rast=True"
            # subprocess.run(command, shell=True)
            # print("3D reconstruction stage1 success")

            # command = f"python dreamgaussian/main2.py --config dreamgaussian/configs/image.yaml input={input_img_path} save_path={NAME} elevation={Elevation} force_cuda_rast=True"
            # subprocess.run(command, shell=True)
            # print("3D reconstruction stage2 success")

            # # 3D obj

            # command = f"python -m kiui.render logs/{NAME}.obj --save_video {NAME}.mp4 --wogui --force_cuda_rast"
            # subprocess.run(command, shell=True)
            # print("3D onj generation success")
            # print("time :", time.time() - start)
