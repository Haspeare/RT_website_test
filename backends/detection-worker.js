// detection-worker.js - Web Worker for Object Detection
// This worker runs in a separate thread to handle CPU-intensive detection tasks

// Import ONNX Runtime
try {
    importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js');
    console.log('[Worker] ONNX Runtime loaded successfully');

    // Configure WASM paths to use local files
    ort.env.wasm.wasmPaths = '/libs/';
    console.log('[Worker] WASM paths configured to:', ort.env.wasm.wasmPaths);
} catch (error) {
    console.error('[Worker] Failed to load ONNX Runtime:', error);
    self.postMessage({
        type: 'init_complete',
        success: false,
        error: 'Failed to load ONNX Runtime: ' + error.message
    });
}

// Worker state
let model = null;
let detector = null;
let tracker = null;

// Traffic flow statistics
let trafficFlowStats = {
    AREA1: { totalCount: 0, totalCars: 0, totalMotorcycles: 0, crossedVehicles: new Set() },
    AREA2: { totalCount: 0, totalCars: 0, totalMotorcycles: 0, crossedVehicles: new Set() },
    AREA3: { totalCount: 0, totalCars: 0, totalMotorcycles: 0, crossedVehicles: new Set() },
    AREA4: { totalCount: 0, totalCars: 0, totalMotorcycles: 0, crossedVehicles: new Set() }
};

let statisticsStartTime = null;

// Counting lines for each lane - defined as line segments
// Format: { x1, y1, x2, y2 } - line from point (x1,y1) to point (x2,y2)
// 計數線定義為線段，由兩個端點定義，可支援任意角度
let countingLines = {
    AREA1: { x1: 369, y1: 337, x2: 222, y2: 409 },  // 基隆->市府向的計數線
    AREA2: { x1: 369, y1: 337, x2: 222, y2: 409 },  // 市府->基隆向的計數線（與AREA1共用）
    AREA3: { x1: 388, y1: 432, x2: 235, y2: 325 },  // 松高路西向的計數線（垂直線範例）
    AREA4: { x1: 388, y1: 432, x2: 235, y2: 325 }   // 松高路東向的計數線（垂直線範例）
};

// ================= TrafficDetector Class (same as in app.js) =================
class TrafficDetector {
    constructor() {
        this.model = null;
        this.modelType = 'onnx';
        this.trafficStats = {
            total_detections: 0,
            car_count: 0,
            motorcycle_count: 0
        };

        // Road edge points
        this.roadEdgePoints = [
            [2, 139],  [287, 317],  [287, 317],[611, 208],
            [2, 195],  [202, 355],  [202, 355],[2, 441],
            [426, 392],[639, 523],  [426, 392],[604, 295],
            [3, 611],  [305, 443],  [305, 442],[576, 636]
        ];

        // Define 4 lane areas
        this.laneAreas = [
            {
                name: 'AREA1',
                color: 'rgba(255, 0, 0, 0.1)',
                polygon: [
                    [287, 317], [611, 208], [557, 268], [369, 337],
                    [222, 409], [3, 516], [2, 441], [202, 355], [287, 317]
                ] //基準線1：[369, 337],[222, 409]
            },
            {
                name: 'AREA2',
                color: 'rgba(0, 255, 0, 0.1)',
                polygon: [
                    [3, 611], [305, 443], [426, 392], [604, 295],
                    [557, 268], [369, 337], [222, 409], [3, 516],
                    [305, 443], [426, 392]
                ] //基準線1：[369, 337],[222, 409]
            },
            {
                name: 'AREA3',
                color: 'rgba(0, 0, 255, 0.1)',
                polygon: [
                    [2, 139], [287, 317], [426, 392], [639, 523],
                    [638, 597], [388, 432], [235, 325], [2, 154],
                    [287, 317], [426, 392]
                ] //基準線2：[388, 432], [235, 325]
            },
            {
                name: 'AREA4',
                color: 'rgba(255, 255, 0, 0.1)',
                polygon: [
                    [2, 195], [202, 355], [305, 442], [576, 636],
                    [638, 597], [388, 432], [235, 325], [2, 154],
                    [202, 355], [305, 442]
                ] //基準線2：[388, 432], [235, 325]
            }
        ];

        // Create simplified polygons
        this.lanePolygons = this.laneAreas.map(area => {
            const uniquePoints = [];
            const pointSet = new Set();
            for (const [x, y] of area.polygon) {
                const key = `${x},${y}`;
                if (!pointSet.has(key)) {
                    pointSet.add(key);
                    uniquePoints.push([x, y]);
                }
            }
            return { name: area.name, color: area.color, polygon: uniquePoints };
        });

        // Pre-calculate bounding boxes
        this.laneBoundingBoxes = this.lanePolygons.map(lane => {
            const xs = lane.polygon.map(p => p[0]);
            const ys = lane.polygon.map(p => p[1]);
            return {
                name: lane.name,
                minX: Math.min(...xs),
                maxX: Math.max(...xs),
                minY: Math.min(...ys),
                maxY: Math.max(...ys)
            };
        });

        // Lane base angles
        this.laneBaseAngles = {
            'AREA1': 154, 'AREA2': 35, 'AREA3': -145, 'AREA4': -27
        };

        // Lane change rules
        this.laneChangeRules = {
            'AREA1': { right_turn: 'AREA3', left_turn: 'AREA4' },
            'AREA2': { right_turn: 'AREA4', left_turn: 'AREA3' },
            'AREA3': { right_turn: 'AREA2', left_turn: 'AREA1' },
            'AREA4': { right_turn: 'AREA1', left_turn: 'AREA2' }
        };

        this.turnAngleThreshold = 30;
        this.roadPolygon = this.createCombinedRoadPolygon();
    }

    createCombinedRoadPolygon() {
        const uniquePoints = [];
        const pointSet = new Set();
        for (const area of this.laneAreas) {
            for (const [x, y] of area.polygon) {
                const key = `${x},${y}`;
                if (!pointSet.has(key)) {
                    pointSet.add(key);
                    uniquePoints.push([x, y]);
                }
            }
        }
        const centroidX = uniquePoints.reduce((sum, p) => sum + p[0], 0) / uniquePoints.length;
        const centroidY = uniquePoints.reduce((sum, p) => sum + p[1], 0) / uniquePoints.length;
        uniquePoints.sort((a, b) => {
            const angleA = Math.atan2(a[1] - centroidY, a[0] - centroidX);
            const angleB = Math.atan2(b[1] - centroidY, b[0] - centroidX);
            return angleA - angleB;
        });
        return uniquePoints;
    }

    isPointInPolygon(x, y, polygon) {
        let inside = false;
        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i][0], yi = polygon[i][1];
            const xj = polygon[j][0], yj = polygon[j][1];
            const intersect = ((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }
        return inside;
    }

    isPointInBoundingBox(x, y, bbox) {
        return x >= bbox.minX && x <= bbox.maxX && y >= bbox.minY && y <= bbox.maxY;
    }

    getVehicleLane(bbox) {
        const [x1, y1, x2, y2] = bbox;
        const centerX = (x1 + x2) / 2;
        const centerY = (y1 + y2) / 2;
        const bottomCenterX = (x1 + x2) / 2;
        const bottomCenterY = y2;

        for (let i = 0; i < this.lanePolygons.length; i++) {
            const lane = this.lanePolygons[i];
            const bbox = this.laneBoundingBoxes[i];
            const bottomInBBox = this.isPointInBoundingBox(bottomCenterX, bottomCenterY, bbox);
            const centerInBBox = this.isPointInBoundingBox(centerX, centerY, bbox);
            if (!bottomInBBox && !centerInBBox) continue;
            if ((bottomInBBox && this.isPointInPolygon(bottomCenterX, bottomCenterY, lane.polygon)) ||
                (centerInBBox && this.isPointInPolygon(centerX, centerY, lane.polygon))) {
                return lane.name;
            }
        }
        return null;
    }

    getAngleDifference(angle1, angle2) {
        let diff = angle2 - angle1;
        while (diff > 180) diff -= 360;
        while (diff < -180) diff += 360;
        return diff;
    }

    getTurnDirection(vehicleAngle, currentLane) {
        if (!currentLane || !this.laneBaseAngles[currentLane]) return 'straight';
        const baseAngle = this.laneBaseAngles[currentLane];
        const angleDiff = this.getAngleDifference(baseAngle, vehicleAngle);
        if (angleDiff <= -this.turnAngleThreshold) return 'left_turn';
        else if (angleDiff >= this.turnAngleThreshold) return 'right_turn';
        else return 'straight';
    }

    determineLaneChange(currentLane, vehicleAngle, bbox) {
        if (!currentLane) return null;
        const turnDirection = this.getTurnDirection(vehicleAngle, currentLane);
        if (turnDirection === 'straight') return null;
        const rules = this.laneChangeRules[currentLane];
        if (!rules) return null;
        const targetLane = rules[turnDirection];
        if (!targetLane) return null;
        const isInTargetLane = this.getVehicleLane(bbox) === targetLane;
        if (isInTargetLane) return targetLane;
        return null;
    }

    async loadModel() {
        try {
            console.log('[Worker] Loading ONNX model...');

            // Try different model paths
            const modelPaths = [
                '/mvt_rtdetr.onnx',
                '../mvt_rtdetr.onnx',
                './mvt_rtdetr.onnx'
            ];

            let lastError = null;
            for (const modelPath of modelPaths) {
                try {
                    console.log(`[Worker] Trying model path: ${modelPath}`);
                    this.model = await ort.InferenceSession.create(modelPath, {
                        executionProviders: ['wasm'],  // Use WASM in worker (WebGL not available)
                        graphOptimizationLevel: 'all'
                    });
                    console.log(`[Worker] Model loaded successfully from: ${modelPath}`);
                    return true;
                } catch (err) {
                    console.warn(`[Worker] Failed to load from ${modelPath}:`, err.message);
                    lastError = err;
                    continue;
                }
            }

            throw lastError || new Error('Failed to load model from any path');

        } catch (error) {
            console.error('[Worker] Failed to load model:', error);
            return false;
        }
    }

    async detectObjects(imageData, confidenceThreshold = 0.5) {
        if (!this.model) {
            console.error('[Worker] Model not loaded');
            return [];
        }

        try {
            const { data: pixels, width, height } = imageData;
            const inputArray = new Float32Array(1 * 3 * 640 * 640);

            for (let i = 0; i < 640 * 640; i++) {
                inputArray[i] = pixels[i * 4] / 255.0;
                inputArray[640 * 640 + i] = pixels[i * 4 + 1] / 255.0;
                inputArray[640 * 640 * 2 + i] = pixels[i * 4 + 2] / 255.0;
            }

            const inputTensor = new ort.Tensor('float32', inputArray, [1, 3, 640, 640]);
            const origSizesTensor = new ort.Tensor('int64', new BigInt64Array([640n, 640n]), [1, 2]);

            const feeds = {
                'images': inputTensor,
                'orig_target_sizes': origSizesTensor
            };

            const inferenceStartTime = performance.now();
            const results = await this.model.run(feeds);
            const inferenceEndTime = performance.now();
            const inferenceTime = (inferenceEndTime - inferenceStartTime).toFixed(2);

            console.log(`[Worker] Inference time: ${inferenceTime}ms`);
            this.lastInferenceTime = inferenceTime;

            const outputs = this.parseOutputs(results);
            if (!outputs) return [];

            const { boxes, labels, scores } = outputs;
            const detections = [];
            const labelsArray = Array.from(labels).map(l => Number(l));
            const maxLabel = Math.max(...labelsArray);
            const use01 = maxLabel <= 1;
            const numItems = Math.min(scores.length, labels.length, boxes.length / 4);

            for (let i = 0; i < numItems; i++) {
                const conf = scores[i];
                if (conf < confidenceThreshold) continue;

                const cls = Number(labels[i]);
                const bx1 = boxes[i * 4];
                const by1 = boxes[i * 4 + 1];
                const bx2 = boxes[i * 4 + 2];
                const by2 = boxes[i * 4 + 3];

                const x1 = Math.max(0, Math.min(Math.round(bx1), 640 - 1));
                const y1 = Math.max(0, Math.min(Math.round(by1), 640 - 1));
                const x2 = Math.max(0, Math.min(Math.round(bx2), 640 - 1));
                const y2 = Math.max(0, Math.min(Math.round(by2), 640 - 1));

                if (x1 >= x2 || y1 >= y2) continue;

                let className;
                if (cls === 3) {
                    className = 'motorcycle';
                } else {
                    className = use01 ? (cls === 0 ? 'motorcycle' : 'car') : (cls === 1 ? 'motorcycle' : 'car');
                }

                const lane = this.getVehicleLane([x1, y1, x2, y2]);
                const isInRoad = lane !== null;

                detections.push({
                    class_id: cls,
                    class_name: className,
                    confidence: conf,
                    bbox: [x1, y1, x2, y2],
                    in_road: isInRoad,
                    lane: lane
                });
            }

            const detections_in_road = detections.filter(d => d.in_road);
            console.log(`[Worker] Detected ${detections.length} objects (${detections_in_road.length} in road)`);
            return detections_in_road;

        } catch (error) {
            console.error('[Worker] Detection error:', error);
            return [];
        }
    }

    parseOutputs(results) {
        try {
            const outputNames = Object.keys(results);
            let boxes = results.boxes || results[outputNames[1]];
            let labels = results.labels || results[outputNames[2]];
            let scores = results.scores || results[outputNames[0]];

            const squeezeBatch = (tensor) => {
                const data = tensor.data;
                const dims = tensor.dims;
                if (dims.length >= 3 && dims[0] === 1) return Array.from(data);
                if (dims.length === 2 && dims[0] === 1) return Array.from(data);
                return Array.from(data);
            };

            if (boxes) boxes = squeezeBatch(boxes);
            if (labels) labels = squeezeBatch(labels);
            if (scores) scores = squeezeBatch(scores);

            if (!boxes || !labels || !scores) {
                console.error('[Worker] Cannot parse model output');
                return null;
            }

            return { boxes, labels, scores };
        } catch (error) {
            console.error('[Worker] Output parsing error:', error);
            return null;
        }
    }
}

// ================= SimpleTracker Class =================
class SimpleTracker {
    constructor(detector) {
        this.detector = detector;
        this.tracks = [];
        this.nextId = 1;
        this.maxAge = 5;
    }

    calculateIoU(box1, box2) {
        const [x1_1, y1_1, x2_1, y2_1] = box1;
        const [x1_2, y1_2, x2_2, y2_2] = box2;
        const xi1 = Math.max(x1_1, x1_2);
        const yi1 = Math.max(y1_1, y1_2);
        const xi2 = Math.min(x2_1, x2_2);
        const yi2 = Math.min(y2_1, y2_2);
        const interArea = Math.max(0, xi2 - xi1) * Math.max(0, yi2 - yi1);
        const box1Area = (x2_1 - x1_1) * (y2_1 - y1_1);
        const box2Area = (x2_2 - x1_2) * (y2_2 - y1_2);
        const unionArea = box1Area + box2Area - interArea;
        return interArea / unionArea;
    }

    getBottomCenter(bbox) {
        const [x1, y1, x2, y2] = bbox;
        return { x: (x1 + x2) / 2, y: y2 };
    }

    calculateAngle(p1, p2) {
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        return Math.atan2(dy, dx) * (180 / Math.PI);
    }

    update(detections, timestamp = Date.now()) {
        const matched = [];

        detections.forEach(det => {
            let bestMatch = null;
            let bestIoU = 0.3;

            this.tracks.forEach(track => {
                const iou = this.calculateIoU(det.bbox, track.bbox);
                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestMatch = track;
                }
            });

            if (bestMatch) {
                const oldCenter = this.getBottomCenter(bestMatch.bbox);
                const newCenter = this.getBottomCenter(det.bbox);
                const dx = newCenter.x - oldCenter.x;
                const dy = newCenter.y - oldCenter.y;
                const timeDelta = (timestamp - bestMatch.timestamp) / 1000;
                const velocityX = timeDelta > 0 ? dx / timeDelta : 0;
                const velocityY = timeDelta > 0 ? dy / timeDelta : 0;
                const speed = Math.sqrt(velocityX * velocityX + velocityY * velocityY);
                const angle = this.calculateAngle(oldCenter, newCenter);
                const currentLane = det.lane;
                const turnDirection = this.detector.getTurnDirection(angle, currentLane);
                const newLane = this.detector.determineLaneChange(currentLane, angle, det.bbox);

                let laneChangeEvent = null;
                if (newLane && newLane !== bestMatch.currentLane) {
                    laneChangeEvent = {
                        from: bestMatch.currentLane,
                        to: newLane,
                        turnDirection: turnDirection,
                        timestamp: timestamp
                    };
                    console.log(`[Worker] Lane change: ${bestMatch.currentLane} → ${newLane}`);
                }

                bestMatch.bbox = det.bbox;
                bestMatch.age = 0;
                bestMatch.timestamp = timestamp;
                bestMatch.velocity = { x: velocityX, y: velocityY };
                bestMatch.speed = speed;
                bestMatch.angle = angle;
                bestMatch.currentLane = newLane || currentLane;
                bestMatch.turnDirection = turnDirection;
                bestMatch.history.push(newCenter);

                if (bestMatch.history.length > 10) bestMatch.history.shift();
                if (laneChangeEvent) bestMatch.laneChanges.push(laneChangeEvent);

                matched.push({
                    ...det,
                    track_id: bestMatch.id,
                    velocity: { x: velocityX, y: velocityY },
                    speed: speed,
                    angle: angle,
                    lane: bestMatch.currentLane,
                    turnDirection: turnDirection,
                    laneChangeEvent: laneChangeEvent
                });
            } else {
                const center = this.getBottomCenter(det.bbox);
                const newTrack = {
                    id: this.nextId++,
                    bbox: det.bbox,
                    age: 0,
                    timestamp: timestamp,
                    velocity: { x: 0, y: 0 },
                    speed: 0,
                    angle: 0,
                    currentLane: det.lane,
                    turnDirection: 'straight',
                    history: [center],
                    laneChanges: []
                };
                this.tracks.push(newTrack);

                matched.push({
                    ...det,
                    track_id: newTrack.id,
                    velocity: { x: 0, y: 0 },
                    speed: 0,
                    angle: 0,
                    turnDirection: 'straight',
                    laneChangeEvent: null
                });
            }
        });

        this.tracks = this.tracks.filter(track => {
            track.age++;
            return track.age < this.maxAge;
        });

        return matched;
    }
}

// ================= Helper Functions =================

// Line segment intersection detection using cross product
// 使用向量叉乘檢測兩條線段是否相交
function doLineSegmentsIntersect(p1, p2, p3, p4) {
    // Line segment 1: p1 -> p2 (vehicle trajectory)
    // Line segment 2: p3 -> p4 (counting line)

    const dx1 = p2.x - p1.x;
    const dy1 = p2.y - p1.y;
    const dx2 = p4.x - p3.x;
    const dy2 = p4.y - p3.y;

    const denominator = dx1 * dy2 - dy1 * dx2;

    // Lines are parallel or coincident
    if (Math.abs(denominator) < 1e-10) {
        return false;
    }

    const dx3 = p1.x - p3.x;
    const dy3 = p1.y - p3.y;

    const t1 = (dx3 * dy2 - dy3 * dx2) / denominator;
    const t2 = (dx3 * dy1 - dy3 * dx1) / denominator;

    // Check if intersection point is within both line segments
    // t1 and t2 should both be in [0, 1]
    return (t1 >= 0 && t1 <= 1 && t2 >= 0 && t2 <= 1);
}

// Check if vehicle crossed counting line
function checkCountingLineCrossing(track, detection) {
    const lane = detection.lane;
    if (!lane || !countingLines[lane]) return false;

    const line = countingLines[lane];

    // Get previous position from track history
    if (!track.history || track.history.length < 2) return false;

    const prevPos = track.history[track.history.length - 2];
    const currPos = track.history[track.history.length - 1];

    // Define counting line segment
    const lineStart = { x: line.x1, y: line.y1 };
    const lineEnd = { x: line.x2, y: line.y2 };

    // Check if vehicle trajectory intersects with counting line
    const crossed = doLineSegmentsIntersect(prevPos, currPos, lineStart, lineEnd);

    return crossed;
}

// Update traffic flow statistics
function updateTrafficFlowStats(trackedDetections) {
    trackedDetections.forEach(det => {
        if (!det.lane || !det.track_id) return;

        const lane = det.lane;
        const vehicleKey = `${det.track_id}-${lane}`;

        // Check if vehicle crossed counting line
        const track = tracker.tracks.find(t => t.id === det.track_id);
        if (track && checkCountingLineCrossing(track, det)) {
            // Only count once per vehicle per lane
            if (!trafficFlowStats[lane].crossedVehicles.has(vehicleKey)) {
                trafficFlowStats[lane].crossedVehicles.add(vehicleKey);
                trafficFlowStats[lane].totalCount++;

                if (det.class_name === 'car') {
                    trafficFlowStats[lane].totalCars++;
                } else if (det.class_name === 'motorcycle') {
                    trafficFlowStats[lane].totalMotorcycles++;
                }

                console.log(`[Worker] 🚦 Vehicle ${det.track_id} crossed ${lane} counting line (Total: ${trafficFlowStats[lane].totalCount})`);
            }
        }
    });
}

// Get traffic flow statistics
function getTrafficFlowStats() {
    const stats = {};
    for (const [lane, data] of Object.entries(trafficFlowStats)) {
        stats[lane] = {
            totalCount: data.totalCount,
            totalCars: data.totalCars,
            totalMotorcycles: data.totalMotorcycles
        };
    }
    return stats;
}

// Calculate statistics duration
function getStatisticsDuration() {
    if (!statisticsStartTime) return 0;
    return Math.floor((Date.now() - statisticsStartTime) / 1000); // seconds
}

// Draw detections on ImageData (Worker-compatible, no canvas needed)
function drawDetectionsOnImageData(imageData, detections, detector) {
    // Create an OffscreenCanvas for drawing (Worker-compatible)
    const canvas = new OffscreenCanvas(imageData.width, imageData.height);
    const ctx = canvas.getContext('2d');

    // Put the original image data on canvas
    ctx.putImageData(imageData, 0, 0);

    // Draw lane areas with semi-transparent fill
    for (const lane of detector.lanePolygons) {
        ctx.fillStyle = lane.color;
        ctx.beginPath();
        if (lane.polygon.length > 0) {
            ctx.moveTo(lane.polygon[0][0], lane.polygon[0][1]);
            for (let i = 1; i < lane.polygon.length; i++) {
                ctx.lineTo(lane.polygon[i][0], lane.polygon[i][1]);
            }
            ctx.closePath();
            ctx.fill();
        }
    }

    // Draw detection boxes (without track ID on image)
    detections.forEach((det) => {
        const [x1, y1, x2, y2] = det.bbox;
        const className = det.class_name;

        // Set color
        if (className === 'car') {
            ctx.strokeStyle = '#FF0000';  // Red
        } else if (className === 'motorcycle') {
            ctx.strokeStyle = '#00FF00';  // Green
        } else {
            ctx.strokeStyle = '#FFFF00';  // Yellow
        }

        // Draw bounding box only (no ID label)
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Note: track_id is still included in detection data for backend
    });

    // Draw counting lines for each lane
    const laneColors = {
        AREA1: '#FF00FF',  // Magenta for AREA1
        AREA2: '#00FFFF',  // Cyan for AREA2
        AREA3: '#FFFF00',  // Yellow for AREA3
        AREA4: '#FF8800'   // Orange for AREA4
    };

    Object.entries(countingLines).forEach(([laneName, line]) => {
        ctx.strokeStyle = laneColors[laneName] || '#FFFFFF';
        ctx.lineWidth = 3;
        ctx.setLineDash([10, 5]);  // Dashed line pattern

        ctx.beginPath();
        ctx.moveTo(line.x1, line.y1);
        ctx.lineTo(line.x2, line.y2);
        ctx.stroke();

        ctx.setLineDash([]);  // Reset to solid line

        // Draw small circles at endpoints for visibility
        ctx.fillStyle = laneColors[laneName] || '#FFFFFF';
        ctx.beginPath();
        ctx.arc(line.x1, line.y1, 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(line.x2, line.y2, 4, 0, 2 * Math.PI);
        ctx.fill();
    });

    // Get the drawn image data back
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

// Calculate lane statistics
function calculateLaneStats(detections) {
    const laneStats = {
        AREA1: { cars: 0, motorcycles: 0, total: 0 },
        AREA2: { cars: 0, motorcycles: 0, total: 0 },
        AREA3: { cars: 0, motorcycles: 0, total: 0 },
        AREA4: { cars: 0, motorcycles: 0, total: 0 }
    };

    detections.forEach(det => {
        if (det.lane && laneStats[det.lane]) {
            laneStats[det.lane].total++;
            if (det.class_name === 'car') {
                laneStats[det.lane].cars++;
            } else if (det.class_name === 'motorcycle') {
                laneStats[det.lane].motorcycles++;
            }
        }
    });

    return laneStats;
}

// ================= Worker Message Handler =================
self.onmessage = async function(e) {
    const { type, data } = e.data;

    switch (type) {
        case 'init':
            try {
                console.log('[Worker] Initializing detector...');

                // Check if ort is available
                if (typeof ort === 'undefined') {
                    throw new Error('ONNX Runtime not loaded in worker context');
                }

                detector = new TrafficDetector();
                console.log('[Worker] TrafficDetector instance created');

                const loaded = await detector.loadModel();
                if (loaded) {
                    tracker = new SimpleTracker(detector);
                    console.log('[Worker] Tracker instance created');

                    // Initialize statistics start time
                    statisticsStartTime = Date.now();
                    console.log('[Worker] Statistics timer started');

                    self.postMessage({
                        type: 'init_complete',
                        success: true
                    });
                } else {
                    self.postMessage({
                        type: 'init_complete',
                        success: false,
                        error: 'Model loading failed'
                    });
                }
            } catch (error) {
                console.error('[Worker] Initialization error:', error);
                self.postMessage({
                    type: 'init_complete',
                    success: false,
                    error: error.message
                });
            }
            break;

        case 'start_statistics':
            // Start/restart statistics
            statisticsStartTime = Date.now();
            console.log('[Worker] 📊 Statistics started');
            self.postMessage({
                type: 'statistics_started',
                success: true
            });
            break;

        case 'reset_statistics':
            // Reset all statistics
            trafficFlowStats = {
                AREA1: { totalCount: 0, totalCars: 0, totalMotorcycles: 0, crossedVehicles: new Set() },
                AREA2: { totalCount: 0, totalCars: 0, totalMotorcycles: 0, crossedVehicles: new Set() },
                AREA3: { totalCount: 0, totalCars: 0, totalMotorcycles: 0, crossedVehicles: new Set() },
                AREA4: { totalCount: 0, totalCars: 0, totalMotorcycles: 0, crossedVehicles: new Set() }
            };
            statisticsStartTime = Date.now();
            console.log('[Worker] 🔄 Statistics reset');
            self.postMessage({
                type: 'statistics_reset',
                success: true
            });
            break;

        case 'update_counting_lines':
            // Update counting line coordinates
            if (data.countingLines) {
                countingLines = data.countingLines;
                console.log('[Worker] 📏 Counting lines updated:', countingLines);
                self.postMessage({
                    type: 'counting_lines_updated',
                    success: true
                });
            }
            break;

        case 'detect':
            if (!detector || !detector.model) {
                self.postMessage({
                    type: 'detection_result',
                    success: false,
                    error: 'Model not loaded'
                });
                return;
            }

            try {
                const { imageData } = data;
                console.log('[Worker] Starting detection...');

                // Step 1: Detect objects
                const detections = await detector.detectObjects(imageData, 0.5);
                console.log(`[Worker] Detected ${detections.length} objects`);

                // Step 2: Track objects
                const trackedDetections = tracker ? tracker.update(detections, Date.now()) : detections;
                console.log(`[Worker] Tracked ${trackedDetections.length} objects`);

                // Step 3: Update traffic flow statistics (counting line crossing)
                updateTrafficFlowStats(trackedDetections);

                // Step 4: Draw detections on canvas (in worker thread)
                console.log('[Worker] Drawing detections on canvas...');
                const annotatedImageData = drawDetectionsOnImageData(imageData, trackedDetections, detector);

                // Step 5: Calculate lane statistics (current count)
                const laneStats = calculateLaneStats(trackedDetections);

                // Step 6: Get traffic flow statistics (cumulative count)
                const trafficFlow = getTrafficFlowStats();
                const statisticsDuration = getStatisticsDuration();

                console.log('[Worker] ✅ Detection and rendering complete');

                // Return annotated image and stats
                self.postMessage({
                    type: 'detection_result',
                    success: true,
                    annotatedImageData: annotatedImageData,  // Already drawn!
                    detections: trackedDetections,
                    laneStats: laneStats,  // Current count
                    trafficFlow: trafficFlow,  // Cumulative count
                    statisticsDuration: statisticsDuration,  // Duration in seconds
                    inferenceTime: detector.lastInferenceTime
                }, [annotatedImageData.data.buffer]); // Transfer buffer for performance

            } catch (error) {
                console.error('[Worker] Detection failed:', error);
                self.postMessage({
                    type: 'detection_result',
                    success: false,
                    error: error.message
                });
            }
            break;

        default:
            console.warn('[Worker] Unknown message type:', type);
    }
};

console.log('[Worker] Detection worker script loaded');
