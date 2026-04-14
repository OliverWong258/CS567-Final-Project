import os
from PIL import Image
from tqdm import tqdm

def convert_box(size, box):
    # size: (w, h), box: [left, top, width, height]
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2] / 2.0) * dw
    y = (box[1] + box[3] / 2.0) * dh
    w = box[2] * dw
    h = box[3] * dh
    return (x, y, w, h)

def convert_visdrone(root_path, split):
    # 定义路径
    img_path = os.path.join(root_path, f'VisDrone2019-DET-{split}', 'images')
    anno_path = os.path.join(root_path, f'VisDrone2019-DET-{split}', 'annotations')
    
    # 创建输出目录
    output_label_path = os.path.join(root_path, 'labels', split)
    output_img_path = os.path.join(root_path, 'images', split)
    os.makedirs(output_label_path, exist_ok=True)
    os.makedirs(output_img_path, exist_ok=True)

    for txt_file in tqdm(os.listdir(anno_path), desc=f"Converting {split}"):
        if not txt_file.endswith('.txt'):
            continue
        
        # 读取对应图片获取宽高
        img_name = txt_file.replace('.txt', '.jpg')
        img_file = os.path.join(img_path, img_name)
        if not os.path.exists(img_file):
            continue
            
        with Image.open(img_file) as img:
            width, height = img.size

        # 转换标注
        with open(os.path.join(anno_path, txt_file), 'r') as f:
            lines = f.readlines()
            
        out_txt_path = os.path.join(output_label_path, txt_file)
        with open(out_txt_path, 'w') as f_out:
            for line in lines:
                data = line.strip().split(',')
                # VisDrone 类别：0-ignored, 1-pedestrian, 2-people, 3-bicycle, 4-car...
                # 通常忽略 0 类别 (背景部分)，类别索引减 1 使其从 0 开始 (pedestrian 为 0)
                cls = int(data[5])
                if cls == 0 or cls == 11:
                    continue
                
                # 调整为 0 索引
                yolo_cls = cls - 1 
                
                box = [float(data[0]), float(data[1]), float(data[2]), float(data[3])]
                bb = convert_box((width, height), box)
                f_out.write(f"{yolo_cls} {' '.join([f'{a:.6f}' for a in bb])}\n")
        
        os.link(img_file, os.path.join(output_img_path, img_name))

# 运行转换
root = "VisDrone" 
for s in ['train', 'val', 'test-dev']:
    convert_visdrone(root, s)