# app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 載入 ONNX 模型
MODEL_PATH = 'model.onnx'
session = ort.InferenceSession(MODEL_PATH)

# 獲取模型輸入輸出資訊
print("=" * 60)
print("模型資訊:")
print("=" * 60)
for inp in session.get_inputs():
    print(f"輸入 - 名稱: {inp.name}, 形狀: {inp.shape}, 類型: {inp.type}")
for out in session.get_outputs():
    print(f"輸出 - 名稱: {out.name}, 形狀: {out.shape}, 類型: {out.type}")
print("=" * 60)

# RT-DETR COCO 類別名稱
COCO_CLASSES = [
     'car', 'motorcycle'
]

def preprocess_image(image_pil, target_size=640):
    """
    RT-DETR 圖片前處理
    """
    # 轉換為 numpy array
    image = np.array(image_pil.convert('RGB'))
    original_h, original_w = image.shape[:2]
    
    # 調整大小 - RT-DETR 通常直接 resize 而不填充
    image_resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # 轉換為模型輸入格式 (1, 3, H, W)
    image_input = image_resized.transpose(2, 0, 1)  # HWC to CHW
    image_input = np.expand_dims(image_input, axis=0).astype(np.float32)
    
    # 歸一化到 0-1
    image_input = image_input / 255.0
    
    # 創建 orig_target_sizes: [batch_size, 2] 格式為 [height, width]
    orig_target_sizes = np.array([[original_h, original_w]], dtype=np.int64)
    
    return image_input, orig_target_sizes, (original_h, original_w)

def parse_onnx_output(outputs, original_shape, conf_threshold=0.5):
    """
    解析 RT-DETR ONNX 模型輸出
    
    RT-DETR 輸出已經是原始圖片座標，不需要額外轉換！
    
    Args:
        outputs: ONNX 模型輸出
        original_shape: 原始圖片尺寸 (height, width)
        conf_threshold: 信心度閾值
    
    Returns:
        檢測結果列表
    """
    # RT-DETR 輸出格式 (可能有所不同，需要根據實際情況調整):
    # outputs[0]: labels [batch_size, num_queries]
    # outputs[1]: boxes [batch_size, num_queries, 4]  
    # outputs[2]: scores [batch_size, num_queries]
    
    # 或者可能是:
    # outputs[0]: boxes [batch_size, num_queries, 4]
    # outputs[1]: scores [batch_size, num_queries, num_classes]
    
    print(f"輸出數量: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  輸出 {i}: shape={out.shape}, dtype={out.dtype}")
    
    # 根據實際輸出調整
    if len(outputs) == 2:
        # 格式1: [boxes, scores]
        boxes = outputs[0][0]     # [num_queries, 4]
        scores = outputs[1][0]    # [num_queries, num_classes] or [num_queries]
        
        if len(scores.shape) == 2:
            # scores 是 [num_queries, num_classes]
            max_scores = np.max(scores, axis=1)
            labels = np.argmax(scores, axis=1)
        else:
            # scores 是 [num_queries]
            max_scores = scores
            labels = np.zeros(len(scores), dtype=np.int32)
            
    elif len(outputs) == 3:
        # 格式2: [labels, boxes, scores]
        labels = outputs[0][0]    # [num_queries]
        boxes = outputs[1][0]     # [num_queries, 4]
        max_scores = outputs[2][0]  # [num_queries]
    else:
        raise ValueError(f"未預期的輸出數量: {len(outputs)}")
    
    results = []
    orig_h, orig_w = original_shape
    
    for i in range(len(max_scores)):
        score = float(max_scores[i])
        
        # 過濾低信心度
        if score < conf_threshold:
            continue
        
        label = int(labels[i])
        box = boxes[i]
        
        # RT-DETR 輸出格式可能是:
        # 1. [x1, y1, x2, y2] - 已經是原始座標
        # 2. [x_center, y_center, width, height] - 歸一化或絕對座標
        
        # 假設是 [x1, y1, x2, y2] 格式（最常見）
        x1, y1, x2, y2 = box
        
        # 如果座標是歸一化的 (0-1)，需要乘以圖片尺寸
        if x2 <= 1.0 and y2 <= 1.0:
            x1 = x1 * orig_w
            y1 = y1 * orig_h
            x2 = x2 * orig_w
            y2 = y2 * orig_h
        
        # 限制在圖片範圍內
        x1 = max(0, min(x1, orig_w))
        y1 = max(0, min(y1, orig_h))
        x2 = max(0, min(x2, orig_w))
        y2 = max(0, min(y2, orig_h))
        
        # 過濾無效框
        if x2 <= x1 or y2 <= y1:
            continue
        
        # 獲取類別名稱
        class_name = COCO_CLASSES[label-2] #if 0 <= label < len(COCO_CLASSES) else f'{label}'
        
        results.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'score': score,
            'class': label,
            'class_name': class_name
        })
    
    # 按信心度排序
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...), conf_threshold: float = 0.5):
    """
    物件偵測 API
    """
    try:
        # 讀取圖片
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        # 前處理
        image_input, orig_target_sizes, original_shape = preprocess_image(image_pil)
        
        print(f"圖片輸入 shape: {image_input.shape}")
        print(f"原始尺寸: {orig_target_sizes}")
        
        # ONNX 推論 - 提供兩個輸入
        input_feed = {
            'images': image_input,
            'orig_target_sizes': orig_target_sizes
        }
        
        outputs = session.run(None, input_feed)
        
        # 後處理
        results = parse_onnx_output(outputs, original_shape, conf_threshold)
        
        return JSONResponse(content={
            "success": True,
            "detections": results,
            "count": len(results),
            "image_size": {
                "width": original_shape[1],
                "height": original_shape[0]
            }
        })
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"錯誤詳情:\n{error_detail}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False, 
                "error": str(e),
                "detail": error_detail
            }
        )

@app.get("/")
async def root():
    return {
        "message": "RT-DETR 物件偵測 API",
        "model": MODEL_PATH,
        "classes": len(COCO_CLASSES),
        "endpoints": {
            "detect": "/detect",
            "classes": "/classes"
        }
    }

@app.get("/classes")
async def get_classes():
    """獲取所有類別"""
    return {
        "classes": COCO_CLASSES,
        "count": len(COCO_CLASSES)
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("🚀 啟動 RT-DETR 物件偵測 API")
    print("=" * 60)
    print(f"📦 模型路徑: {MODEL_PATH}")
    print(f"🎯 支援類別: {len(COCO_CLASSES)} 個 COCO 類別")
    print(f"🌐 API 位址: http://localhost:8000")
    print(f"📖 API 文檔: http://localhost:8000/docs")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)