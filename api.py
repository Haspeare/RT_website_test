# api.py - 車流辨識系統 FastAPI 服務
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from PIL import Image
import io
import base64
import time
from app import TrafficDetector, StreamCapture

app = FastAPI(title="車流辨識系統", version="1.0.0")

# 允許跨域請求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 靜態檔案服務
app.mount("/static", StaticFiles(directory="."), name="static")

# 初始化檢測器和串流截圖
detector = TrafficDetector()
stream_capture = StreamCapture()

# ================= FastAPI Endpoints =================
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "model_type": detector.model_type
    }

@app.post("/detect")
async def detect_image(file: UploadFile = File(...), confidence: float = 0.5):
    if detector.model is None:
        raise HTTPException(status_code=503, detail="模型未載入")
    try:
        image_bytes = await file.read()
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        detections = detector.detect_objects(image_pil, confidence)
        annotated_image = detector.draw_detections(image_pil, detections)
        img_buffer = io.BytesIO()
        annotated_image.save(img_buffer, format='JPEG')
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        return {
            "success": True,
            "detections": detections,
            "detection_count": len(detections),
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
            "confidence_threshold": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"檢測失敗: {str(e)}")

@app.get("/capture_stream")
async def capture_stream(confidence: float = 0.5, reload: bool = False):
    """截取監視器串流並進行檢測"""
    if detector.model is None:
        raise HTTPException(status_code=503, detail="模型未載入")

    try:
        print(f"🎥 截取監視器串流，信心度: {confidence}, 重新載入: {reload}")

        stream_image = stream_capture.capture_stream(force_reload=reload)

        if stream_image is None:
            print("⚠️ 串流截圖失敗")
            return {
                "success": False,
                "error": "截圖失敗"
            }

        # 直接將整個畫面 resize 成 640x640
        print(f"📐 原始截圖尺寸: {stream_image.size}")
        stream_image = stream_image.resize((640, 640), Image.Resampling.LANCZOS)
        print(f"📏 調整後尺寸: {stream_image.size}")

        # 進行檢測
        detections = detector.detect_objects(stream_image, confidence)

        # 統計結果
        car_count = sum(1 for d in detections if d["class_name"] == "car")
        motorcycle_count = sum(1 for d in detections if d["class_name"] == "motorcycle")

        # 計算平均信心度
        if detections:
            avg_confidence = sum(d["confidence"] for d in detections) / len(detections)
        else:
            avg_confidence = 0.0

        # 繪製檢測結果
        annotated_image = detector.draw_detections(stream_image, detections)

        # 轉換為 base64
        img_buffer = io.BytesIO()
        annotated_image.save(img_buffer, format='JPEG', quality=75)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()

        return {
            "success": True,
            "car_count": car_count,
            "motorcycle_count": motorcycle_count,
            "total_count": car_count + motorcycle_count,
            "confidence_threshold": confidence,
            "avg_confidence": avg_confidence,
            "timestamp": time.time(),
            "annotated_image": f"data:image/jpeg;base64,{img_base64}"
        }

    except Exception as e:
        print(f"❌ 串流檢測失敗: {e}")
        raise HTTPException(status_code=500, detail=f"串流檢測失敗: {str(e)}")

@app.get("/stats")
async def get_stats():
    return {
        "traffic_stats": dict(detector.traffic_stats),
        "model_type": detector.model_type
    }

# ================= 啟動 =================
if __name__ == "__main__":
    print("啟動車流辨識系統...")
    print("系統將在 http://localhost:5000 啟動")
    uvicorn.run(app, host="0.0.0.0", port=5000)
