# app.py - 車流辨識系統 (簡化穩定版)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import cv2
import numpy as np
from PIL import Image
import io
import base64
import time
import random
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

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

# ================= TrafficDetector =================
class TrafficDetector:
    def __init__(self):
        self.model = None
        self.model_type = "onnx"
        self.traffic_stats = defaultdict(int)
        self.load_model()

    def load_model(self):
        """載入 ONNX 模型"""
        print("=" * 50)
        print("正在載入車流辨識模型...")
        print("=" * 50)

        onnx_path = "model.onnx"
        if os.path.exists(onnx_path):
            try:
                import onnxruntime as ort
                self.model = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                print("✅ 成功載入 ONNX 模型!")
                print(f"模型輸入: {[inp.name for inp in self.model.get_inputs()]}")
                print(f"模型輸出: {[out.name for out in self.model.get_outputs()]}")
            except Exception as e:
                print(f"❌ ONNX 模型載入失敗: {e}")
                self.model = None
        else:
            print(f"❌ 未找到模型檔案: {onnx_path}")
            self.model = None

        print("=" * 50)

    def detect_objects(self, image, confidence_threshold=0.5):
        """檢測物件"""
        if self.model is None:
            return []

        try:
            # 轉換影像
            if isinstance(image, Image.Image):
                frame = np.array(image)
            else:
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w = frame.shape[:2]

            # 前處理
            input_image = cv2.resize(frame, (640, 640))
            input_image = input_image.astype(np.float32) / 255.0
            input_image = input_image.transpose(2, 0, 1)
            input_image = np.expand_dims(input_image, axis=0)

            # 準備輸入
            inputs = {}
            for inp in self.model.get_inputs():
                if inp.name == "images":
                    inputs["images"] = input_image
                elif inp.name == "orig_target_sizes":
                    inputs["orig_target_sizes"] = np.array([[640, 640]], dtype=np.int64)

            # 模型推理
            outputs = self.model.run(None, inputs)

            # 解析輸出
            try:
                out_infos = self.model.get_outputs()
                name_to_output = {out_infos[i].name: outputs[i] for i in range(len(out_infos))}

                def squeeze_batch(arr):
                    arr_np = np.array(arr)
                    if arr_np.ndim >= 3 and arr_np.shape[0] == 1:
                        return arr_np[0]
                    if arr_np.ndim == 2 and arr_np.shape[0] == 1:
                        return arr_np[0]
                    return arr_np

                boxes = name_to_output.get("boxes")
                labels = name_to_output.get("labels")
                scores = name_to_output.get("scores")

                if boxes is not None:
                    boxes = squeeze_batch(boxes)
                if labels is not None:
                    labels = squeeze_batch(labels)
                if scores is not None:
                    scores = squeeze_batch(scores)

                if boxes is None or labels is None or scores is None:
                    if len(outputs) >= 3:
                        if boxes is None:
                            boxes = squeeze_batch(outputs[1])
                        if labels is None:
                            labels = squeeze_batch(outputs[2])
                        if scores is None:
                            scores = squeeze_batch(outputs[0])
                    else:
                        return []
            except Exception as parse_error:
                print(f"輸出解析錯誤: {parse_error}")
                return []

            detections = []
            num_items = min(len(scores), len(labels), boxes.shape[0])

            try:
                use_01 = int(np.max(labels)) <= 1
            except Exception:
                use_01 = True

            scale_x = w / 640.0
            scale_y = h / 640.0
            for i in range(num_items):
                conf = float(scores[i])
                if conf < confidence_threshold:
                    continue

                cls = int(labels[i])
                bx1, by1, bx2, by2 = map(float, boxes[i])
                x1 = int(max(0, min(bx1 * scale_x, w - 1)))
                y1 = int(max(0, min(by1 * scale_y, h - 1)))
                x2 = int(max(0, min(bx2 * scale_x, w - 1)))
                y2 = int(max(0, min(by2 * scale_y, h - 1)))
                if x1 >= x2 or y1 >= y2:
                    continue

                if cls == 3:
                    class_name = "motorcycle"
                else:
                    if use_01:
                        class_name = {0: "motorcycle", 1: "car"}.get(cls, f"class_{cls}")
                    else:
                        class_name = {1: "motorcycle", 2: "car"}.get(cls, f"class_{cls}")

                detections.append({
                    "class_id": cls,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

            print(f"檢測到 {len(detections)} 個物件")
            return detections

        except Exception as e:
            print(f"檢測錯誤: {e}")
            return []

    def draw_detections(self, image, detections):
        try:
            if isinstance(image, Image.Image):
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                frame = image.copy()
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                class_name = det["class_name"]
                conf = det["confidence"]
                if class_name == "car":
                    color = (0, 0, 255)
                elif class_name == "motorcycle":
                    color = (0, 255, 0)
                else:
                    color = (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"繪製檢測結果錯誤: {e}")
            return image

# 串流截圖類
class StreamCapture:
    def __init__(self):
        self.driver = None
        self.setup_driver()
    
    def setup_driver(self):
        """設置 Chrome 瀏覽器"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1250")  

            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            print("✅ Chrome 瀏覽器初始化成功")
        except Exception as e:
            print(f"❌ Chrome 瀏覽器初始化失敗: {e}")
            self.driver = None
    
    def capture_stream(self, force_reload=False):
        """截取串流畫面（僅截取影片元素）"""
        if not self.driver:
            return None

        try:
            # 如果是強制重新載入或首次載入，重新取得頁面
            if force_reload or not hasattr(self, '_page_loaded'):
                print("🔄 重新載入串流頁面...")
                self.driver.get("https://hls.bote.gov.taipei/live/index.html?id=139")
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                time.sleep(1)
                self._page_loaded = True

            # 嘗試定位並截取影片元素
            try:
                # 等待影片元素載入
                video_element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, "video"))
                )
                print("✅ 找到影片元素")

                # 滾動到影片元素，確保完整可見
                self.driver.execute_script("arguments[0].scrollIntoView(true);", video_element)
                time.sleep(0.1)

                # 使用 JavaScript 設定影片元素尺寸為 1889x1259 並確保播放
                self.driver.execute_script("""
                    var video = document.getElementById('video');
                    if (video) {
                        // 設定影片尺寸
                        video.style.width = '1889px';
                        video.style.height = '1259px';
                        video.width = 1889;
                        video.height = 1259;

                        // 確保影片播放
                        if (video.paused) {
                            video.play();
                        }
                    }
                """)
                time.sleep(0.1)

                # 獲取影片元素的尺寸
                size = video_element.size
                location = video_element.location
                print(f"📐 影片元素尺寸: {size['width']}x{size['height']}, 位置: ({location['x']}, {location['y']})")

                # 截取影片元素
                screenshot = video_element.screenshot_as_png
                image = Image.open(io.BytesIO(screenshot))
                print(f"✅ 截圖成功，尺寸: {image.size}")
                return image

            except Exception as video_error:
                print(f"⚠️ 無法定位影片元素: {video_error}")
                print("📸 改為截取整個頁面")
                # 如果找不到影片元素，則截取整個頁面
                screenshot = self.driver.get_screenshot_as_png()
                image = Image.open(io.BytesIO(screenshot))
                return image

        except Exception as e:
            print(f"❌ 串流截圖失敗: {e}")
            return None
    
    def close(self):
        """關閉瀏覽器"""
        if self.driver:
            self.driver.quit()

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