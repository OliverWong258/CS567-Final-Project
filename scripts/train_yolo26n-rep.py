from ultralytics import YOLO
import os

# 在检测头前加入RepConv后的YOLO26n
def train_visdrone():
    # 1. 加载模型配置
    # 这里使用 yaml 加载架构，并用 .pt 加载预训练权重
    model = YOLO("yolo26n-rep.yaml").load("yolo26n.pt") 

    # 2. 开始训练
    results = model.train(
        # 数据集路径
        data="/root/autodl-tmp/CS567-Final-Project/datasets/VisDrone/visdrone.yaml",
        
        # 训练参数
        epochs=200,             # 训练轮数
        patience=50,            # 早停机制：50轮无提升则停止
        imgsz=640,              # 输入尺寸
        batch=128,               # 批量大小
        
        # 硬件配置
        device='0,1',           
        workers=4,              # 数据加载线程数
        
        optimizer='MuSGD',      
        amp=True,               # 开启自动混合精度训练
        
        project='VisDrone_YOLO26n',
        name='yolo26n_rep_exp2',
        exist_ok=True,          
        
        save=True,              # 保存模型
        plots=True,             # 生成曲线图
    )

if __name__ == "__main__":
    train_visdrone()