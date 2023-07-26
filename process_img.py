from PIL import Image, ImageStat
import os
import numpy as np
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

    # img_dir = '/data/data_wbw/data/cropped_tyre/006163123012900_3/test/'
    # img_file = 'A011141165_0.jpg'
    # img = Image.open(os.path.join(img_dir, img_file))
    # img = np.array(img)
    # img[500:503, :] = (0,0,0)
    # img = Image.fromarray(np.uint8(img))
    # img.save('/data/data_wbw/data/cropped_tyre/006163123012900_3/test/1.jpg')

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
    
    img_dir = '/data2/chen/uad-tire/1-按规格显著病茨/7.00R16-8PR[CR908+A]朝阳无内 AA/70/'
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')]
    save_dir = '/data/data_wbw/data/cropped_tyre/1-按规格显著病茨/7.00R16-8PR[CR908+A]朝阳无内 AA/70_2000/'
    val_dir = '/data/data_wbw/data/cropped_tyre/1-按规格显著病茨/7.00R16-8PR[CR908+A]朝阳无内 AA/70_2000/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    col_num = 3
    import random
    random.seed(3)
    index = list(range(0, len(img_files)))
    random.shuffle(index)

    for j in range(0, col_num):
        save_path = os.path.join(save_dir, '{}/'.format(j))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    for j in range(0, col_num):
        val_path = os.path.join(val_dir, '{}/'.format(j))
        if not os.path.exists(val_path):
            os.makedirs(val_path)
    for img_num, img_file in enumerate(img_files):
        img = Image.open(os.path.join(img_dir, img_file))
        if img.size[0] >= 2000:
            continue
        if img.mode == "P":
            img = img.convert('RGB')
        i = 0
        start = 0
        patch_length = img.size[0] / col_num
        while start < img.size[1]:
            for j in range(0, col_num):
                if start + patch_length < img.size[1]:
                    rg = img.crop((patch_length * j, start, patch_length * (j + 1), start + patch_length))
                elif img.size[1] - start > patch_length / 2:
                    rg = img.crop((patch_length * j, img.size[1] - patch_length, patch_length * (j + 1), img.size[1]))
                else:
                    break
                if(index[img_num] < 0.9 * len(img_files)):         
                    rg.save(os.path.join(save_dir, '{}/{}_{}.jpg'.format(j, img_file.split('.')[0], i)))
                else:
                    rg.save(os.path.join(val_dir, '{}/{}_{}.jpg'.format(j, img_file.split('.')[0], i)))
            start += patch_length
            i += 1
        print(img_file)