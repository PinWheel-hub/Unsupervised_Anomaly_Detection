import os, cv2
import numpy as np
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
def main():
    root = 'test_imgs'
    img_dir = os.path.join(root, 'raw_imgs' )
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
    col_num = 3
    for img_num, img_file in enumerate(img_files):
        img = cv2.imread(os.path.join(img_dir, img_file))
        if img.shape[1] > 2000:
            img = cv2.resize(img, (img.shape[1] // 2, int(img.shape[0] // 1.5)), interpolation=cv2.INTER_CUBIC)
        img, _ = remove_white_cols(img) 
        i = 0
        patch_length = int(img.shape[1] // col_num)
        row_num = img.shape[0] // patch_length
        save_dir = os.path.join(root, f'{os.path.splitext(img_file)[0]}')
        save_side_dir = os.path.join(root, f'{os.path.splitext(img_file)[0]}_side')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_side_dir):
            os.makedirs(save_side_dir)
        for i in range(0, row_num):
            for j in range(0, col_num):
                if patch_length * (i + 1) < img.shape[0] - 200:
                    rg = img[patch_length * i: patch_length * (i + 1), patch_length * j: patch_length * (j + 1)]
                # elif img.shape[0] - patch_length * i > patch_length / 2:
                #     rg = img[img.shape[0] - patch_length: img.shape[0], patch_length * j: patch_length * (j + 1)]
                else:
                    break
                cv2.imwrite(os.path.join(save_dir if j == 1 else save_side_dir, f'{os.path.splitext(img_file)[0]}_{j}_{i}.jpg'), rg if j < 2 else cv2.flip(rg, 1))

if __name__ == '__main__':
    main()