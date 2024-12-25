import cv2
import os

def process_folder(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(input_folder) 
                 if os.path.isdir(os.path.join(input_folder, f))]
    
    # 遍历所有子文件夹
    for folder_name in subfolders:
        input_path = os.path.join(input_folder, folder_name)
        output_path = os.path.join(output_folder, folder_name)
        
        # 确保输出子文件夹存在
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # 处理当前文件夹中的所有图片，支持png,jpg,jepg
        print(f"正在处理文件夹: {folder_name}")
        for filename in os.listdir(input_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 目标文件路径
                output_file = os.path.join(output_path, filename)
                # 检查文件是否已处理，处理了就跳过
                if os.path.exists(output_file):
                    continue
                
                # 读取图片
                img_path = os.path.join(input_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # 转换为灰度图
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # 应用Canny边缘检测
                    edges = cv2.Canny(gray, 50, 100)  # 可以调整这些阈值
                    
                    edges = cv2.bitwise_not(edges)
                    
                    # 保存结果
                    cv2.imwrite(output_file, edges)
                    print(f"已处理: {folder_name}/{filename}")
                else:
                    print(f"无法读取图片: {folder_name}/{filename}")
        
        print(f"文件夹 {folder_name} 处理完成")

def main():
    input_folder = r"D:\workspace\data\ABC\obj\blender_new"
    output_folder = r"D:\workspace\data\ABC\obj\edgemap"
    
    process_folder(input_folder, output_folder)
    print("所有文件夹处理完成!")

if __name__ == "__main__":
    main()
