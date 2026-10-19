import cv2
import time
import os
from ultralytics import YOLO

def benchmark_model(model_path, video_path, output_path, is_pt=True):
    print(f"\n正在测试模型: {model_path}")
    
    # 加载模型
    if is_pt:
        model = YOLO(model_path)
    else:
        # ONNX 显式指定 task，防止找不到配置
        model = YOLO(model_path, task='detect')
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 预热阶段
    ret, frame = cap.read()
    if ret:
        for _ in range(5):
            model.predict(frame, conf=0.25, imgsz=640, verbose=False)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_time = 0
    frame_count = 0
    total_objects = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = time.perf_counter()
        
        # 推理
        results = model.predict(frame, conf=0.25, imgsz=640, verbose=False)
        
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        
        # 统计目标数量
        boxes = results[0].boxes
        if boxes is not None:
            total_objects += len(boxes)
        
        # ==========================================
        # 手动绘制检测框，绕过 ONNX 找不到标签的 Bug
        # ==========================================
        annotated_frame = frame.copy()
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # 安全获取类别名，找不到就用 ID 代替
                if hasattr(model, 'names') and isinstance(model.names, dict):
                    name = model.names.get(cls_id, f"id:{cls_id}")
                else:
                    name = f"id:{cls_id}"
                    
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{name} {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, max(y1 - 5, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        out.write(annotated_frame)
        
    cap.release()
    out.release()
    
    avg_time_ms = (total_time / frame_count) * 1000
    print(f"总帧数: {frame_count}")
    print(f"总目标数: {total_objects}")
    print(f"平均单帧推理耗时: {avg_time_ms:.2f} ms")
    print(f"结果已保存至: {output_path}")
    
    return frame_count, total_objects, avg_time_ms


if __name__ == "__main__":
    video = "armor_video.mp4"
    
    # 测试 PT 模型
    pt_frames, pt_objs, pt_time = benchmark_model("best.pt", video, "results/output_pt.mp4", is_pt=True)
    
    # 测试 ONNX 模型
    onnx_frames, onnx_objs, onnx_time = benchmark_model("models/best.onnx", video, "results/output_onnx.mp4", is_pt=False)
    
    print("\n====================================")
    print("--- 最终对比数据 ---")
    print(f"PT 模型   - 平均耗时: {pt_time:.2f} ms, 目标总数: {pt_objs}")
    print(f"ONNX 模型 - 平均耗时: {onnx_time:.2f} ms, 目标总数: {onnx_objs}")
    print("====================================")
