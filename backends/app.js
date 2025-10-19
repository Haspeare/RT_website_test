// app.js - Traffic Detection System Frontend Application (Complete TrafficDetector Class)

// ================= TrafficDetector Class =================
class TrafficDetector {
    constructor() {
        this.model = null;
        this.modelType = 'onnx';
        this.trafficStats = {
            total_detections: 0,
            car_count: 0,
            motorcycle_count: 0
        };

        // Road edge points (from app.py) - 16 points to draw 8 lines
        this.roadEdgePoints = [
            [2, 139],  [287, 317],  // Line 1
            [287, 317],[611, 208],  // Line 2
            [2, 195],  [202, 355],  // Line 3
            [202, 355],[2, 441],  // Line 4
            [426, 392],[639, 523],  // Line 5
            [427, 391],[604, 295],  // Line 6
            [3, 611],  [305, 443],  // Line 7
            [305, 442],[576, 636]  // Line 8
        ];
    }
    // Load ONNX model
    async loadModel() {
        try {
            console.log('='.repeat(50));
            console.log('Loading traffic detection model...');
            console.log('='.repeat(50));

            // Use ONNX Runtime Web
            this.model = await ort.InferenceSession.create('./model.onnx', {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            console.log('Successfully loaded ONNX model!');
            console.log('Model inputs:', this.model.inputNames);
            console.log('Model outputs:', this.model.outputNames);
            console.log('='.repeat(50));

            return true;
        } catch (error) {
            console.error('Failed to load ONNX model:', error);
            this.model = null;
            return false;
        }
    }

    // Detect objects
    async detectObjects(image, confidenceThreshold = 0.5) {
        if (!this.model) {
            console.error('Model not loaded');
            return [];
        }

        try {
            // Prepare input image (resize to 640x640)
            const canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 640;
            const ctx = canvas.getContext('2d');

            // Draw and resize image
            ctx.drawImage(image, 0, 0, 640, 640);

            // Get image data
            const imageData = ctx.getImageData(0, 0, 640, 640);
            const pixels = imageData.data;

            // Preprocessing: Convert to model input format [1, 3, 640, 640]
            // Normalize to 0-1 and convert to RGB channel order
            const inputArray = new Float32Array(1 * 3 * 640 * 640);

            for (let i = 0; i < 640 * 640; i++) {
                inputArray[i] = pixels[i * 4] / 255.0;                    // R channel
                inputArray[640 * 640 + i] = pixels[i * 4 + 1] / 255.0;   // G channel
                inputArray[640 * 640 * 2 + i] = pixels[i * 4 + 2] / 255.0; // B channel
            }

            // Prepare model input
            const inputTensor = new ort.Tensor('float32', inputArray, [1, 3, 640, 640]);
            const origSizesTensor = new ort.Tensor('int64', new BigInt64Array([640n, 640n]), [1, 2]);

            const feeds = {
                'images': inputTensor,
                'orig_target_sizes': origSizesTensor
            };

            // Model inference
            const inferenceStartTime = performance.now();
            const results = await this.model.run(feeds);
            const inferenceEndTime = performance.now();
            const inferenceTime = (inferenceEndTime - inferenceStartTime).toFixed(2);

            console.log(`   🧠 Model inference time: ${inferenceTime}ms`);

            // Store inference time for reporting
            this.lastInferenceTime = inferenceTime;

            // Parse output
            const outputs = this.parseOutputs(results);
            if (!outputs) {
                return [];
            }

            const { boxes, labels, scores } = outputs;

            // Convert detection results
            const detections = [];

            // Since image is already 640x640, no scaling needed
            // But keep for compatibility if image size differs
            const scaleX = image.width / 640.0;
            const scaleY = image.height / 640.0;

            // Determine label format (convert BigInt to Number)
            const labelsArray = Array.from(labels).map(l => Number(l));
            const maxLabel = Math.max(...labelsArray);
            const use01 = maxLabel <= 1;

            const numItems = Math.min(scores.length, labels.length, boxes.length / 4);

            console.log(`Processing ${numItems} detections, scaleX=${scaleX}, scaleY=${scaleY}`);

            for (let i = 0; i < numItems; i++) {
                const conf = scores[i];

                if (conf < confidenceThreshold) continue;

                const cls = Number(labels[i]);
                const bx1 = boxes[i * 4];
                const by1 = boxes[i * 4 + 1];
                const bx2 = boxes[i * 4 + 2];
                const by2 = boxes[i * 4 + 3];

                // Direct coordinates (no scaling since image is 640x640)
                const x1 = Math.max(0, Math.min(Math.round(bx1), 640 - 1));
                const y1 = Math.max(0, Math.min(Math.round(by1), 640 - 1));
                const x2 = Math.max(0, Math.min(Math.round(bx2), 640 - 1));
                const y2 = Math.max(0, Math.min(Math.round(by2), 640 - 1));

                if (x1 >= x2 || y1 >= y2) continue;

                // Class name mapping (same as Python version)
                let className;
                if (cls === 3) {
                    className = 'motorcycle';
                } else {
                    if (use01) {
                        className = cls === 0 ? 'motorcycle' : 'car';
                    } else {
                        className = cls === 1 ? 'motorcycle' : 'car';
                    }
                }

                detections.push({
                    class_id: cls,
                    class_name: className,
                    confidence: conf,
                    bbox: [x1, y1, x2, y2]
                });
            }

            console.log(`Detected ${detections.length} objects`);
            return detections;

        } catch (error) {
            console.error('Detection error:', error);
            return [];
        }
    }

    // Parse model outputs
    parseOutputs(results) {
        try {
            const outputNames = Object.keys(results);

            let boxes = results.boxes || results[outputNames[1]];
            let labels = results.labels || results[outputNames[2]];
            let scores = results.scores || results[outputNames[0]];

            // Extract data and remove batch dimension
            const squeezeBatch = (tensor) => {
                const data = tensor.data;
                const dims = tensor.dims;

                if (dims.length >= 3 && dims[0] === 1) {
                    return Array.from(data);
                }
                if (dims.length === 2 && dims[0] === 1) {
                    return Array.from(data);
                }
                return Array.from(data);
            };

            if (boxes) boxes = squeezeBatch(boxes);
            if (labels) labels = squeezeBatch(labels);
            if (scores) scores = squeezeBatch(scores);

            if (!boxes || !labels || !scores) {
                console.error('Cannot parse model output');
                return null;
            }

            return { boxes, labels, scores };

        } catch (error) {
            console.error('Output parsing error:', error);
            return null;
        }
    }

    // Draw detections
    drawDetections(image, detections) {
        try {
            const canvas = document.createElement('canvas');
            canvas.width = image.width;
            canvas.height = image.height;
            const ctx = canvas.getContext('2d');

            // Draw original image
            ctx.drawImage(image, 0, 0);

            // Draw road edges
            this.drawRoadEdges(ctx);

            // Draw detection boxes (only outline)
            detections.forEach((det, index) => {
                const [x1, y1, x2, y2] = det.bbox;
                const className = det.class_name;

                console.log(`Drawing detection ${index + 1}: ${className} at [${x1}, ${y1}, ${x2}, ${y2}]`);

                // Set color (BGR -> RGB conversion)
                if (className === 'car') {
                    ctx.strokeStyle = '#FF0000';  // Red
                } else if (className === 'motorcycle') {
                    ctx.strokeStyle = '#00FF00';  // Green
                } else {
                    ctx.strokeStyle = '#FFFF00';  // Yellow
                }

                // Draw bounding box outline only
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            });

            return canvas;

        } catch (error) {
            console.error('Drawing detection results error:', error);
            return null;
        }
    }

    // Draw road edges
    drawRoadEdges(ctx) {
        try {
            // Check if the number of coordinate points is correct (should be 16 points)
            if (this.roadEdgePoints.length !== 16) {
                console.warn(`Road edge points count incorrect: ${this.roadEdgePoints.length}, should be 16`);
                return;
            }

            // Road edge line color (BGR 255, 255, 0 -> RGB #00FFFF cyan)
            ctx.strokeStyle = '#00FFFF';  // Cyan
            ctx.lineWidth = 2;

            // Draw a line for every two points (8 lines total)
            for (let i = 0; i < 16; i += 2) {
                const [x1, y1] = this.roadEdgePoints[i];
                const [x2, y2] = this.roadEdgePoints[i + 1];

                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
            }

        } catch (error) {
            console.error('Drawing road edges error:', error);
        }
    }

    // Update statistics
    updateStats(detections) {
        this.trafficStats.total_detections = detections.length;
        this.trafficStats.car_count = detections.filter(d => d.class_name === 'car').length;
        this.trafficStats.motorcycle_count = detections.filter(d => d.class_name === 'motorcycle').length;
    }
}

// ================= Global Variables =================
let detector = null;
let isDetecting = false;
let detectionInterval = null;
let isFirstDetection = true;  // Track if this is the first detection

// ================= Initialization =================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Application initializing...');

    // Create detector instance
    detector = new TrafficDetector();

    // Load ONNX model
    const loaded = await detector.loadModel();

    if (!loaded) {
        alert('Model loading failed, please confirm model.onnx file exists');
    }

    // Setup event listeners
    setupEventListeners();

    console.log('Application initialized successfully');
});

// ================= Setup Event Listeners =================
function setupEventListeners() {
    // Sidebar menu
    const menuLinks = document.querySelectorAll('.menu-link');
    menuLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const pageId = link.getAttribute('data-page');
            switchPage(pageId);

            // Update menu state
            menuLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });

    // Detect button
    const detectBtn = document.getElementById('detectBtn');
    if (detectBtn) {
        detectBtn.addEventListener('click', toggleDetection);
    }
}

// ================= Page Switching =================
function switchPage(pageId) {
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => page.classList.remove('active'));

    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        targetPage.classList.add('active');
    }
}

// ================= Start/Stop Detection =================
async function toggleDetection() {
    const detectBtn = document.getElementById('detectBtn');

    if (!detector || !detector.model) {
        alert('Model not loaded, please refresh the page');
        return;
    }

    if (!isDetecting) {
        // Start detection
        isDetecting = true;
        detectBtn.textContent = '⏸️ Stop Detection';
        detectBtn.style.background = 'linear-gradient(135deg, #f44336 0%, #e91e63 100%)';

        console.log('Starting real-time detection');

        // Execute immediately once
        await performDetection();

        // Execute detection every 500ms (0.5 seconds) for real-time monitoring
        detectionInterval = setInterval(performDetection, 500);
    } else {
        // Stop detection
        isDetecting = false;
        detectBtn.textContent = '🔍 Start Detection';
        detectBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';

        if (detectionInterval) {
            clearInterval(detectionInterval);
            detectionInterval = null;
        }

        // Reset first detection flag for next time
        isFirstDetection = true;

        console.log('Detection stopped');
    }
}

// ================= Perform Detection =================
async function performDetection() {
    if (!detector || !detector.model) {
        console.error('Detector not initialized');
        return;
    }

    try {
        console.log('Capturing stream frame...');

        // Show loading animation only on first detection
        const loading = document.getElementById('loading');
        if (loading && isFirstDetection) {
            loading.style.display = 'block';
        }

        // Get stream snapshot from backend
        const response = await fetch('/capture_frame');
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Capture failed');
        }

        console.log('Stream capture successful');

        // Load image
        const image = await loadImage(data.image);
        console.log(`Image loaded: ${image.width}x${image.height}`);

        // Perform object detection (using TrafficDetector)
        const detections = await detector.detectObjects(image, 0.5);

        console.log(`Detected ${detections.length} objects:`, detections);

        // Draw detection results (using TrafficDetector)
        const annotatedCanvas = detector.drawDetections(image, detections);
        console.log('Drawing complete, canvas:', annotatedCanvas ? `${annotatedCanvas.width}x${annotatedCanvas.height}` : 'null');

        if (annotatedCanvas) {
            // Display results
            displayResults(annotatedCanvas, detections);

            // Update statistics
            detector.updateStats(detections);

            // Update statistics to server (include inference time)
            await updateStatsToServer(detections, detector.lastInferenceTime);
        }

        // Hide loading animation and mark first detection as complete
        if (loading && isFirstDetection) {
            loading.style.display = 'none';
            isFirstDetection = false;  // Only show loading animation once
        }

    } catch (error) {
        console.error('Detection failed:', error);

        const loading = document.getElementById('loading');
        if (loading) loading.style.display = 'none';

        // Don't show error message if in continuous detection mode
        if (!isDetecting) {
            alert(`Detection failed: ${error.message}`);
        }
    }
}

// ================= Load Image =================
function loadImage(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('Image loading failed'));
        img.src = src;
    });
}

// ================= Display Results =================
function displayResults(canvas, detections) {
    // Display detection image
    const detectionImage = document.getElementById('detectionImage');
    const noImagePlaceholder = document.getElementById('noImagePlaceholder');

    if (detectionImage && noImagePlaceholder) {
        detectionImage.src = canvas.toDataURL('image/jpeg', 0.9);
        detectionImage.style.display = 'block';
        noImagePlaceholder.style.display = 'none';
    }

    // Calculate statistics
    const carCount = detections.filter(d => d.class_name === 'car').length;
    const motorcycleCount = detections.filter(d => d.class_name === 'motorcycle').length;
    const totalCount = detections.length;
    const avgConfidence = totalCount > 0
        ? (detections.reduce((sum, d) => sum + d.confidence, 0) / totalCount * 100).toFixed(1)
        : 0;

    // Update main page statistics
    const totalDetectionsEl = document.getElementById('totalDetections');
    const totalCarsEl = document.getElementById('totalCars');
    const totalMotorcyclesEl = document.getElementById('totalMotorcycles');
    const avgConfidenceEl = document.getElementById('avgConfidence');

    if (totalDetectionsEl) totalDetectionsEl.textContent = totalCount;
    if (totalCarsEl) totalCarsEl.textContent = carCount;
    if (totalMotorcyclesEl) totalMotorcyclesEl.textContent = motorcycleCount;
    if (avgConfidenceEl) avgConfidenceEl.textContent = avgConfidence + '%';

    // Update backend data page
    updateBackendDataPage(detections, carCount, motorcycleCount, avgConfidence);
}

// ================= Update Backend Data Page =================
function updateBackendDataPage(detections, carCount, motorcycleCount, avgConfidence) {
    const resultsDiv = document.getElementById('results');
    if (!resultsDiv) return;

    resultsDiv.style.display = 'block';

    resultsDiv.innerHTML = `
        <h3>Detection Statistics</h3>
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">${detections.length}</div>
                <div class="stat-label">Total Detections</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">${carCount}</div>
                <div class="stat-label">Cars</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">${motorcycleCount}</div>
                <div class="stat-label">Motorcycles</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">${avgConfidence}%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
        </div>

        <h3 style="margin-top: 30px;">Detection Details</h3>
        <div class="detection-list">
            ${detections.map((det, index) => `
                <div class="detection-item">
                    <strong>${det.class_name === 'car' ? 'Car' : 'Motorcycle'} #${index + 1}</strong>
                    <div>Position: (${det.bbox[0]}, ${det.bbox[1]}) - (${det.bbox[2]}, ${det.bbox[3]})</div>
                    <div class="confidence">Confidence: ${(det.confidence * 100).toFixed(1)}%</div>
                </div>
            `).join('')}
        </div>
    `;
}

// ================= Update Statistics to Server =================
async function updateStatsToServer(detections, inferenceTime) {
    try {
        await fetch('/update_stats', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                detections,
                inferenceTime: parseFloat(inferenceTime)
            })
        });
    } catch (error) {
        console.error('Failed to update statistics:', error);
    }
}
