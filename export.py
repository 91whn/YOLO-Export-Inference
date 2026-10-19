from ultralytics import YOLO
import os
import shutil

def main():
    pt_model_path = "best.pt"
    if not os.path.exists(pt_model_path):
        print("未找到 best.pt 模型，请确保文件在当前目录下")
        return

    print("--- 开始导出模型 ---")
    model = YOLO(pt_model_path)
    
    # 导出为 ONNX 格式
    export_path = model.export(format='onnx', imgsz=640, dynamic=False)
    print(f"导出成功，原始生成路径: {export_path}")

    os.makedirs("models", exist_ok=True)
    target_path = os.path.join("models", "best.onnx")
    
    # 移动文件
    if os.path.exists(target_path):
        os.remove(target_path)
    shutil.move(export_path, target_path)
    print(f"模型已移动到: {target_path}")

    # 打印大小对比
    pt_size = os.path.getsize(pt_model_path) / (1024 * 1024)
    onnx_size = os.path.getsize(target_path) / (1024 * 1024)
    print(f"--- 模型大小对比 ---")
    print(f"原始 .pt 模型大小: {pt_size:.2f} MB")
    print(f"导出 .onnx 模型大小: {onnx_size:.2f} MB")

if __name__ == "__main__":
    main()