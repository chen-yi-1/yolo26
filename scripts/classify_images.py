import os
import shutil

def get_image_files(folder):
    """获取文件夹中的所有图片文件"""
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
    images = []
    for f in os.listdir(folder):
        if f.lower().endswith(image_extensions):
            images.append(f)
    return images

def get_class_from_txt(txt_path):
    """从txt文件中获取类别（0或1）"""
    try:
        with open(txt_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line:
                parts = first_line.split()
                if parts:
                    return int(parts[0])
    except Exception:
        return None
    return None

def classify_images():
    base_dir = '/root/yolo26/dataset'
    labels_dir = os.path.join(base_dir, 'labels')
    abnormal_dir = os.path.join(base_dir, 'abnormal')
    healthy_dir = os.path.join(base_dir, 'healthy')
    others_dir = os.path.join(base_dir, 'others')
    
    # 确保目标文件夹存在
    os.makedirs(abnormal_dir, exist_ok=True)
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(others_dir, exist_ok=True)
    
    # 获取所有源文件夹中的图片
    all_images = []
    for folder in [abnormal_dir, healthy_dir, others_dir]:
        all_images.extend([(os.path.join(folder, img), img) for img in get_image_files(folder)])
    
    print(f"发现 {len(all_images)} 张图片")
    
    moved_count = 0
    others_count = 0
    
    for src_path, img_name in all_images:
        # 获取对应的txt文件名
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        txt_path = os.path.join(labels_dir, txt_name)
        
        # 获取类别
        cls = get_class_from_txt(txt_path)
        
        if cls == 0:
            dst_dir = abnormal_dir
        elif cls == 1:
            dst_dir = healthy_dir
        else:
            dst_dir = others_dir
            others_count += 1
        
        # 移动文件（如果不在目标位置）
        dst_path = os.path.join(dst_dir, img_name)
        if src_path != dst_path:
            shutil.move(src_path, dst_path)
            moved_count += 1
    
    print(f"已移动 {moved_count} 张图片")
    print(f"无标签的图片（放入others）: {others_count} 张")
    
    # 统计最终结果
    print("\n最终文件统计：")
    print(f"abnormal: {len(get_image_files(abnormal_dir))} 张")
    print(f"healthy: {len(get_image_files(healthy_dir))} 张")
    print(f"others: {len(get_image_files(others_dir))} 张")

if __name__ == '__main__':
    classify_images()
