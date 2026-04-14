import time
import torch
from ultralytics import YOLO
import pandas as pd

def benchmark_models_v2(model_names, imgsz=1280, batch_size=32, runs=1000):
    results = []
    device = 0
    
    print(f"正在 5090 上进行压力测试 | 分辨率: {imgsz} | Batch: {batch_size}")
    
    for name in model_names:
        print(f"\n{'='*60}\n模型: {name}")
        
        # 1. 导出引擎
        model = YOLO(name)
        print(f"正在导出 TensorRT (Batch={batch_size})...")
        export_path = model.export(format='engine', imgsz=imgsz, half=True, device=device, batch=batch_size)
        
        # 2. 加载
        trt_model = YOLO(export_path, task='detect')
        
        # 3. 构造测试数据
        input_data = torch.zeros((batch_size, 3, imgsz, imgsz)).to(device).half() # 使用 FP16 数据
        
        # 4. 预热
        print("预热中...")
        for _ in range(20):
            trt_model.predict(input_data, verbose=False)
            
        # 5. 计时
        print(f"开始执行 {runs} 次循环测试...")
        torch.cuda.synchronize()
        t_start = time.time()
        
        for _ in range(runs):
            trt_model.predict(input_data, verbose=False)
            
        torch.cuda.synchronize()
        t_end = time.time()
        
        # 6. 指标换算
        total_time = t_end - t_start
        avg_batch_latency = (total_time / runs) * 1000           # 处理这一组(32张)的时间
        avg_img_latency = avg_batch_latency / batch_size         # 平均每张图的时间
        fps = (runs * batch_size) / total_time                  # 真正的每秒处理帧数
        
        results.append({
            "Model": name,
            "Batch_Size": batch_size,
            "Img_Latency (ms)": f"{avg_img_latency:.3f}",
            "Batch_Latency (ms)": f"{avg_batch_latency:.2f}",
            "FPS": f"{int(fps)}"
        })
        
        del model, trt_model
        torch.cuda.empty_cache()

    # 输出结果
    df = pd.DataFrame(results)
    print("\n" + " 5090 压力测试报告 ")
    print(df.to_markdown(index=False))
    return df

if __name__ == "__main__":
    target_models = ["yolo11n.pt", "yolo11s.pt", "yolo26n.pt", "yolo26s.pt"]
    benchmark_models_v2(target_models)