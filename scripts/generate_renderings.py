import os
import re

def generate_renderings_txt(base_dir):
    """
    Generate renderings.txt for each subdirectory in the base directory.

    Args:
        base_dir (str): The path to the directory containing sample folders (e.g., edgemap).

    Returns:
        None
    """
    # 检查基础目录是否存在
    if not os.path.exists(base_dir):
        print(f"Error: Base directory {base_dir} does not exist.")
        return
    
    # 遍历子文件夹
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        # 确保当前路径是文件夹
        if not os.path.isdir(folder_path):
            continue

        # 找到文件夹中的图片文件
        image_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(('.png', '.jpg', '.jpeg'))  # 支持的图片格式
        ]

        if not image_files:
            print(f"Warning: No image files found in {folder_path}. Skipping.")
            continue

        # 按数字提取并排序
        def extract_view_number(filename):
            match = re.search(r'_view_(\d+)', filename)
            return int(match.group(1)) if match else float('inf')  # 没有数字视为最大值
        
        image_files.sort(key=extract_view_number)

        # 写入renderings.txt
        renderings_txt_path = os.path.join(folder_path, 'renderings.txt')
        with open(renderings_txt_path, 'w') as f:
            f.write('\n'.join(image_files))
        
        print(f"Generated renderings.txt for {folder_path}")

if __name__ == "__main__":
    # 指定edgemap文件夹的路径
    base_directory = "./data/renderingimg/edgemap"
    
    generate_renderings_txt(base_directory)
