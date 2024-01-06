from PIL import Image, ImageStat
import os
import numpy as np
import cv2
if __name__ == '__main__':
    # img_dir = '/data2/chen/spec10k/006163123012900/'
    # img_files = os.listdir(img_dir)
    # for img_file in img_files:
    #     img = Image.open(os.path.join(img_dir, img_file))
    #     i = 0
    #     start = 0
    #     patch_length = img.size[0] / 3
    #     while start < img.size[1]:
    #         if start + patch_length < img.size[1]:
    #             rg = img.crop((patch_length * 1, start, patch_length * 2, start + patch_length))
    #         else:
    #             rg = img.crop((patch_length * 1, img.size[1] - patch_length, patch_length * 2, img.size[1]))
    #         rg.save('/data/data_wbw/data/cropped_tyre/006163123012900_3/train/{0}_{1}.jpg'.format(img_file.split('.')[0], i))
    #         start += patch_length
    #         i += 1
    #     print(img_file)

    # ===================================================================================

    img_dir = '/data/data_wbw/data/cropped_tyre/Tyre_AD/JD2023-GuoChan/test/good'
    img_file = 'Image_20221208095016C1_1_4.jpg'
    img = Image.open(os.path.join(img_dir, img_file))
    img = np.array(img)
    # img[200:204, :] = (0,0,0)
    img = Image.fromarray(np.uint8(img))
    img.save('Image_20221208095016C1_1_4.jpg')

    # ===================================================================================

    # img_dir = '/data2/chen/633/'
    # img_file = 'H811202023.jpg'
    # img = Image.open(os.path.join(img_dir, img_file))
    # patch_length = img.size[0] / 3

    # img = img.crop((patch_length * 1, 0, patch_length * 2, img.size[1]))
    # img = np.array(img)
    # img[2545:2550, 0:1200] = (0,0,0)
    # img[3395:3400, 0:1200] = (0,0,0)
    # img[8195:8200, 0:1200] = (0,0,0)
    # img = Image.fromarray(np.uint8(img))
    # img.save('/data/data_wbw/data/cropped_tyre/{0}'.format(img_file))

    # step = 50
    # i = 0
    # while i * step + patch_length <= 2550:
    #     rg = img.crop((patch_length * 1, i * step, patch_length * 2, i * step + patch_length))
    #     rg.save('/data/data_wbw/data/cropped_tyre/{0}/train/{1}.jpg'.format(img_file.split('.')[0], i))
    #     i += 1
    # j = i
    # while j * step < 3400:
    #     rg = img.crop((patch_length * 1, j * step, patch_length * 2, j * step + patch_length))
    #     rg.save('/data/data_wbw/data/cropped_tyre/{0}/test/{1}.jpg'.format(img_file.split('.')[0], j))
    #     j += 10
    # while j * step + patch_length <= 8200:
    #     rg = img.crop((patch_length * 1, j * step, patch_length * 2, j * step + patch_length))
    #     rg.save('/data/data_wbw/data/cropped_tyre/{0}/train/{1}.jpg'.format(img_file.split('.')[0], i))
    #     j += 1
    #     i += 1

    # ===================================================================================
    # img_type = "2-常用规格/6.50R16-12PR[CR926]朝阳"
    # img_dir = f'/data2/chen/uad-tire/{img_type}/病茨/'
    # other_dirs = [d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d))]
    # img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
    # for d in other_dirs:
    #     img_files.extend([os.path.join(d, f) for f in os.listdir(os.path.join(img_dir, d)) if f.endswith('.jpg') or f.endswith('.png')])
        
    # save_dir = f'/data/data_wbw/data/cropped_tyre/{img_type}/anomaly_2000/'
    # val_dir = f'/data/data_wbw/data/cropped_tyre/{img_type}/anomaly_2000/'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # col_num = 3

    # import random
    # random.seed(3)
    # index = list(range(0, len(img_files)))
    # print(len(img_files))
    # random.shuffle(index)

    # for j in range(0, col_num):
    #     save_path = os.path.join(save_dir, '{}/'.format(j))
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    # for j in range(0, col_num):
    #     val_path = os.path.join(val_dir, '{}/'.format(j))
    #     if not os.path.exists(val_path):
    #         os.makedirs(val_path)
    # for img_num, img_file in enumerate(img_files):
    #     img = Image.open(os.path.join(img_dir, img_file))
    #     if img.size[0] > 2000: # or img.size[0] > 4000:
    #         continue
    #     if img.mode == "P":
    #         img = img.convert('RGB')
    #     i = 0
    #     start = 0
    #     patch_length = img.size[0] / col_num
    #     while start < img.size[1]:
    #         for j in range(0, col_num):
    #             if start + patch_length < img.size[1]:
    #                 rg = img.crop((patch_length * j, start, patch_length * (j + 1), start + patch_length))
    #             elif img.size[1] - start > patch_length / 2:
    #                 rg = img.crop((patch_length * j, img.size[1] - patch_length, patch_length * (j + 1), img.size[1]))
    #             else:
    #                 break
    #             if(index[img_num] < 0.9 * len(img_files)):         
    #                 rg.save(os.path.join(save_dir, '{}/{}_{}.jpg'.format(j, os.path.basename(img_file).split('.')[0], i)))
    #             else:
    #                 rg.save(os.path.join(val_dir, '{}/{}_{}.jpg'.format(j, os.path.basename(img_file).split('.')[0], i)))
    #         start += patch_length
    #         i += 1
    #     print(img_file)

    def remove_white_cols(img_cv):
            # 将图像转换为灰度图
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

            # 获取图像列的平均亮度
            col_means = np.mean(gray, axis=0)
            threshold = np.mean(col_means)
            mask = col_means < threshold
            # 找到第一个和最后一个非白色列的索引
            first_col = np.argmax(mask)

            # 找到最后一个 True 的位置
            last_col = len(mask) - 1 - np.argmax(np.flip(mask))

            # 从原始图像中裁剪除去白色条纹的部分
            result = img_cv[:, first_col: last_col]

            return result, first_col
    
    # img_dir = f'/data2/chen/JD2023-GuoChan'
    # img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
    # save_dir = f'/data/data_wbw/data/cropped_tyre/JD2023-GuoChan'
    # save_side_dir = f'/data/data_wbw/data/cropped_tyre/JD2023-GuoChan_side'
    # train_dir = os.path.join(save_dir, 'train')
    # test_dir = os.path.join(save_dir, 'test')
    # train_side_dir = os.path.join(save_side_dir, 'train')
    # test_side_dir = os.path.join(save_side_dir, 'test')
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # if not os.path.exists(save_side_dir):
    #     os.makedirs(save_side_dir)
    # if not os.path.exists(train_dir):
    #     os.makedirs(train_dir)
    # if not os.path.exists(test_dir):
    #     os.makedirs(test_dir)
    # if not os.path.exists(train_side_dir):
    #     os.makedirs(train_side_dir)
    # if not os.path.exists(test_side_dir):
    #     os.makedirs(test_side_dir)
    # col_num = 3
    # import random
    # from tqdm import tqdm
    # random.seed(3)
    # index = list(range(0, len(img_files)))
    # random.shuffle(index)
    # for img_num, img_file in tqdm(enumerate(img_files), total=len(img_files)):
    #     img = cv2.imread(os.path.join(img_dir, img_file))
    #     if img.shape[1] > 2000:
    #         img = cv2.resize(img, (img.shape[1] // 2, int(img.shape[0] // 1.5)), interpolation=cv2.INTER_CUBIC)
    #     img, _ = remove_white_cols(img) 
    #     i = 0
    #     patch_length = int(img.shape[1] // col_num)
    #     row_num = img.shape[0] // patch_length
    #     for i in range(0, row_num):
    #         for j in range(0, col_num):
    #             if patch_length * (i + 1) < img.shape[0] - 200:
    #                 rg = img[patch_length * i: patch_length * (i + 1), patch_length * j: patch_length * (j + 1)]
    #             # elif img.shape[0] - patch_length * i > patch_length / 2:
    #             #     rg = img[img.shape[0] - patch_length: img.shape[0], patch_length * j: patch_length * (j + 1)]
    #             else:
    #                 break
    #             if index[img_num] < 0.9 * len(img_files):
    #                 cv2.imwrite(os.path.join(train_dir if j == 1 else train_side_dir, f'{os.path.splitext(img_file)[0]}_{j}_{i}.jpg'), rg if j < 2 else cv2.flip(rg, 1))
    #             else:
    #                 cv2.imwrite(os.path.join(test_dir if j == 1 else test_side_dir, f'{os.path.splitext(img_file)[0]}_{j}_{i}.jpg'), rg if j < 2 else cv2.flip(rg, 1))