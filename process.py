import os
import fitz  # PyMuPDF

def pdf_to_png(folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(folder_path, filename)
            
            try:
                # 打开PDF文件
                doc = fitz.open(pdf_path)
                
                # 遍历每一页
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # 设置分辨率 (DPI)
                    zoom = 4  # 缩放因子 (4倍相当于约300 DPI)
                    matrix = fitz.Matrix(zoom, zoom)
                    
                    # 将PDF页面转换为图片
                    pix = page.get_pixmap(matrix=matrix)
                    
                    # 生成输出文件名 (原始名称_页码.png)
                    output_name = f"{os.path.splitext(filename)[0]}_page{page_num+1}.png"
                    output_path = os.path.join(folder_path, output_name)
                    
                    # 保存为PNG
                    pix.save(output_path)
                    print(f"已转换: {filename} 第{page_num+1}页 -> {output_name}")
                
                doc.close()
            except Exception as e:
                print(f"处理 {filename} 时出错: {str(e)}")

if __name__ == "__main__":
    # 设置包含PDF的文件夹路径
    target_folder = r"D:\2025\RUO-DD\data\img\distilled_img"  # 修改为你的文件夹路径
    pdf_to_png(target_folder)