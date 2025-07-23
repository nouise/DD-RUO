import cv2
import os
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
import io  # 添加这行导入

def pdf_to_image(pdf_path, dpi=300):
    """
    Convert PDF first page to high-quality image
    
    Args:
        pdf_path (str): Path to PDF file
        dpi (int): Resolution for PDF rendering
    
    Returns:
        numpy.ndarray: Image array in BGR format for OpenCV
    """
    try:
        # 打开PDF文件
        doc = fitz.open(pdf_path)
        page = doc[0]  # 获取第一页
        
        # 设置缩放矩阵以获得高分辨率
        zoom = dpi / 72  # PDF默认72 DPI
        mat = fitz.Matrix(zoom, zoom)
        
        # 渲染页面为图像
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        
        # 转换为PIL图像
        pil_img = Image.open(io.BytesIO(img_data))
        
        # 转换为OpenCV格式 (BGR)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        doc.close()
        return cv_img
        
    except Exception as e:
        print(f"转换PDF失败 {pdf_path}: {e}")
        return None

def create_video_from_pdf_pairs(input_dir, output_path, fps=1, frame_duration=3, dpi=300, 
                               left_max_height=1800, right_width_scale=1.0, right_height_scale=1.0):
    """
    Create MP4 video from PDF pairs, showing tm and full PDFs side by side
    
    Args:
        input_dir (str): Directory containing PDF pairs
        output_path (str): Output MP4 file path
        fps (int): Frames per second for the video
        frame_duration (int): Duration in seconds for each pair
        dpi (int): DPI for PDF rendering (higher = better quality)
        left_max_height (int): Fixed max height for left image
        right_width_scale (float): Width scaling factor for right image (relative to left)
        right_height_scale (float): Height scaling factor for right image (relative to left)
    """
    
    # PDF对配置
    pdf_pairs = [
        ("tm_imagefruit.pdf", "imagefruit_full.pdf"),
        ("tm_imagemeow.pdf", "imagemeow_full.pdf"),
        ("tm_imagenette.pdf", "imagenette_full.pdf"),
        ("tm_imagesquawk.pdf", "imagesquawk_full.pdf"),
        ("tm_imagewoof.pdf", "imagewoof_full.pdf"),
        ("tm_imageyellow.pdf", "imageyellow_full.pdf")
    ]
    
    # 验证文件是否存在
    valid_pairs = []
    for tm_file, full_file in pdf_pairs:
        tm_path = os.path.join(input_dir, tm_file)
        full_path = os.path.join(input_dir, full_file)
        
        if os.path.exists(tm_path) and os.path.exists(full_path):
            valid_pairs.append((tm_path, full_path))
        else:
            print(f"警告: PDF文件不存在 - {tm_file} 或 {full_file}")
    
    if not valid_pairs:
        print("未找到有效的PDF对")
        return
    
    print(f"使用 {dpi} DPI 渲染PDF...")
    
    # 转换第一对PDF获取尺寸
    first_tm_img = pdf_to_image(valid_pairs[0][0], dpi)
    first_full_img = pdf_to_image(valid_pairs[0][1], dpi)
    
    if first_tm_img is None or first_full_img is None:
        print("错误: 无法转换第一对PDF")
        return
    
    # 获取转换后的图片尺寸
    tm_h, tm_w = first_tm_img.shape[:2]
    full_h, full_w = first_full_img.shape[:2]
    
    print(f"PDF转换后图片尺寸: tm={tm_w}x{tm_h}, full={full_w}x{full_h}")
    
    # 计算左侧图片尺寸（固定高度）
    left_scale = left_max_height / tm_h
    left_width = int(tm_w * left_scale)
    left_height = left_max_height
    
    # 确保左侧尺寸为偶数
    left_width = left_width - (left_width % 2)
    left_height = left_height - (left_height % 2)
    
    # 计算右侧图片尺寸（基于左侧尺寸和缩放系数）
    right_width = int(left_width * right_width_scale)
    right_height = int(left_height * right_height_scale)
    
    # 确保右侧尺寸为偶数
    right_width = right_width - (right_width % 2)
    right_height = right_height - (right_height % 2)
    
    # 合并后的视频尺寸
    video_width = left_width + right_width
    video_height = max(left_height, right_height)
    
    print(f"左侧图片尺寸: {left_width}x{left_height} (缩放比例: {left_scale:.2f})")
    print(f"右侧图片尺寸: {right_width}x{right_height} (宽度系数: {right_width_scale}, 高度系数: {right_height_scale})")
    print(f"视频尺寸: {video_width}x{video_height}")
    
    # 创建视频写入器
    success = False
    output_path_final = output_path
    
    # 优先使用 H.264 编码器生成 MP4 文件
    try:
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
        if video_writer.isOpened():
            print(f"使用 H264 编码器，输出为: {output_path}")
            success = True
            output_path_final = output_path
    except Exception as e:
        print(f"H264 编码器失败: {e}")
    
    # 备用方案1：mp4v
    if not success:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height))
            if video_writer.isOpened():
                print(f"使用 mp4v 编码器，输出为: {output_path}")
                success = True
                output_path_final = output_path
        except Exception as e:
            print(f"mp4v 编码器失败: {e}")
    
    # 备用方案2：MJPG（输出 AVI）
    if not success:
        try:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            output_path_final = output_path.replace('.mp4', '_hq.avi')
            video_writer = cv2.VideoWriter(output_path_final, fourcc, fps, (video_width, video_height))
            if video_writer.isOpened():
                print(f"使用 MJPG 编码器（高画质），输出为: {output_path_final}")
                success = True
        except Exception as e:
            print(f"MJPG 编码器失败: {e}")
    
    if not success:
        print("错误: 无法创建视频文件")
        return
    
    # 处理每个PDF对
    total_pairs = len(valid_pairs)
    for i, (tm_path, full_path) in enumerate(valid_pairs):
        tm_name = os.path.basename(tm_path)
        full_name = os.path.basename(full_path)
        print(f"处理PDF对 {i+1}/{total_pairs}: {tm_name} + {full_name}")
        
        # 转换PDF为图像
        tm_img = pdf_to_image(tm_path, dpi)
        full_img = pdf_to_image(full_path, dpi)
        
        if tm_img is None or full_img is None:
            print(f"跳过无法转换的PDF: {tm_path} 或 {full_path}")
            continue
        
        # 分别调整左右图片尺寸
        tm_resized = cv2.resize(tm_img, (left_width, left_height), interpolation=cv2.INTER_LANCZOS4)
        full_resized = cv2.resize(full_img, (right_width, right_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 创建画布（统一高度）
        canvas = np.full((video_height, video_width, 3), 255, dtype=np.uint8)
        
        # 将左侧图片放置在画布左侧（垂直居中）
        left_y_offset = (video_height - left_height) // 2
        canvas[left_y_offset:left_y_offset + left_height, 0:left_width] = tm_resized
        
        # 将右侧图片放置在画布右侧（垂直居中）
        right_y_offset = (video_height - right_height) // 2
        canvas[right_y_offset:right_y_offset + right_height, left_width:left_width + right_width] = full_resized
        
        # 写入帧
        frames_per_pair = fps * frame_duration
        for frame_idx in range(frames_per_pair):
            video_writer.write(canvas)
    
    # 释放视频写入器
    video_writer.release()
    print(f"视频创建成功: {output_path_final}")
    print(f"总共处理了 {total_pairs} 个PDF对")

def main():
    """Main function"""
    input_directory = r"D:\2025\DD-RUO\figures"  # PDF文件目录
    output_video = "pdf_pairs_video.mp4"
    
    # 创建高质量视频：300 DPI，每对显示3秒
    # left_max_height: 左侧图片固定高度
    # right_width_scale: 右侧图片宽度缩放系数
    # right_height_scale: 右侧图片高度缩放系数
    create_video_from_pdf_pairs(
        input_directory, 
        output_video, 
        fps=1, 
        frame_duration=3, 
        dpi=300,
        left_max_height=1800,
        right_width_scale=1,  # 右侧图片宽度为左侧的1.2倍
        right_height_scale=0.99  # 右侧图片高度为左侧的0.8倍
    )

if __name__ == "__main__":
    main()