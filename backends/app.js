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
            [426, 392],[604, 295],  // Line 6
            [3, 611],  [305, 443],  // Line 7
            [305, 442],[576, 636]  // Line 8
        ];

        // Define 4 lane areas with polygons (vertices in order)
        this.laneAreas = [
            {
                name: 'AREA1',
                color: 'rgba(255, 0, 0, 0.1)',      // Red
                polygon: [
                    [287, 317], [611, 208],         // Line 2 (start -> end)
                    [557, 268],                     // Right edge point
                    [369, 337],                     // Right edge point
                    [222, 409],                     // Bottom edge point
                    [3, 516],                       // Bottom edge point
                    [2, 441], [202, 355],           // Line 4 (end -> start, reversed)
                    [287, 317]                      // Close polygon back to start
                ]
            },
            {
                name: 'AREA2',
                color: 'rgba(0, 255, 0, 0.1)',      // Green
                polygon: [
                    [3, 611], [305, 443],           // Line 7
                    [426, 392], [604, 295],         // Line 6
                    [557, 268], [369, 337],         // Right edge
                    [222, 409], [3, 516],           // Bottom edge
                    [305, 443], [426, 392]          // Connect back
                ]
            },
            {
                name: 'AREA3',
                color: 'rgba(0, 0, 255, 0.1)',      // Blue
                polygon: [
                    [2, 139], [287, 317],           // Line 1
                    [426, 392], [639, 523],         // Line 5
                    [638, 597], [388, 432],         // Bottom edge
                    [235, 325], [2, 154],           // Left edge
                    [287, 317], [426, 392]          // Connect back
                ]
            },
            {
                name: 'AREA4',
                color: 'rgba(255, 255, 0, 0.1)',    // Yellow
                polygon: [
                    [2, 195], [202, 355],           // Line 3
                    [305, 442], [576, 636],         // Line 8
                    [638, 597], [388, 432],         // Bottom edge
                    [235, 325], [2, 154],           // Left edge
                    [202, 355], [305, 442]          // Connect back
                ]
            }
        ];

        // Create simplified polygons for each lane (remove duplicates)
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

            return {
                name: area.name,
                color: area.color,
                polygon: uniquePoints
            };
        });

        // ===== OPTIMIZATION: Pre-calculate bounding boxes for fast filtering =====
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

        // Define base vector angles for each lane area
        this.laneBaseAngles = {
            'AREA1': 154,   // degrees
            'AREA2': 35,    // degrees
            'AREA3': -145,  // degrees
            'AREA4': -27    // degrees
        };

        // Define lane change rules: current_lane -> { left_turn: target_lane, right_turn: target_lane }
        this.laneChangeRules = {
            'AREA1': { right_turn: 'AREA3', left_turn: 'AREA4' },
            'AREA2': { right_turn: 'AREA4', left_turn: 'AREA3' },
            'AREA3': { right_turn: 'AREA2', left_turn: 'AREA1' },
            'AREA4': { right_turn: 'AREA1', left_turn: 'AREA2' }
        };

        this.turnAngleThreshold = 30; // degrees for detecting turns

        // Create combined road polygon from all lane areas
        this.roadPolygon = this.createCombinedRoadPolygon();
    }

    // Create combined road polygon from all lane areas
    createCombinedRoadPolygon() {
        // Extract all unique points from all lane areas
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

        // Sort points to form a polygon (by angle from centroid)
        const centroidX = uniquePoints.reduce((sum, p) => sum + p[0], 0) / uniquePoints.length;
        const centroidY = uniquePoints.reduce((sum, p) => sum + p[1], 0) / uniquePoints.length;

        uniquePoints.sort((a, b) => {
            const angleA = Math.atan2(a[1] - centroidY, a[0] - centroidX);
            const angleB = Math.atan2(b[1] - centroidY, b[0] - centroidX);
            return angleA - angleB;
        });

        return uniquePoints;
    }

    // Check if a point is inside the road polygon using ray casting algorithm
    isPointInRoad(x, y) {
        const polygon = this.roadPolygon;
        let inside = false;

        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i][0], yi = polygon[i][1];
            const xj = polygon[j][0], yj = polygon[j][1];

            const intersect = ((yi > y) !== (yj > y))
                && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }

        return inside;
    }

    // Check if a point is inside a specific polygon
    isPointInPolygon(x, y, polygon) {
        let inside = false;

        for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
            const xi = polygon[i][0], yi = polygon[i][1];
            const xj = polygon[j][0], yj = polygon[j][1];

            const intersect = ((yi > y) !== (yj > y))
                && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
            if (intersect) inside = !inside;
        }

        return inside;
    }

    // ===== OPTIMIZATION: Fast bounding box check =====
    // Check if point is inside a bounding box (much faster than polygon check)
    isPointInBoundingBox(x, y, bbox) {
        return x >= bbox.minX && x <= bbox.maxX && y >= bbox.minY && y <= bbox.maxY;
    }

    // Check which lane area a vehicle is in (if any)
    getVehicleLane(bbox) {
        // bbox format: [x1, y1, x2, y2]
        const [x1, y1, x2, y2] = bbox;

        // Calculate center point and bottom-center of the bounding box
        const centerX = (x1 + x2) / 2;
        const centerY = (y1 + y2) / 2;
        const bottomCenterX = (x1 + x2) / 2;
        const bottomCenterY = y2;

        // ===== OPTIMIZATION: Check bounding box first (fast filter) =====
        // This eliminates ~70% of polygon checks
        for (let i = 0; i < this.lanePolygons.length; i++) {
            const lane = this.lanePolygons[i];
            const bbox = this.laneBoundingBoxes[i];

            // Quick reject: if not in bounding box, skip expensive polygon check
            const bottomInBBox = this.isPointInBoundingBox(bottomCenterX, bottomCenterY, bbox);
            const centerInBBox = this.isPointInBoundingBox(centerX, centerY, bbox);

            if (!bottomInBBox && !centerInBBox) {
                continue; // Skip this lane entirely
            }

            // Only do expensive polygon check if in bounding box
            if ((bottomInBBox && this.isPointInPolygon(bottomCenterX, bottomCenterY, lane.polygon)) ||
                (centerInBBox && this.isPointInPolygon(centerX, centerY, lane.polygon))) {
                return lane.name; // Return lane name (AREA1, AREA2, AREA3, AREA4)
            }
        }

        return null; // Not in any lane
    }

    // Check if a bounding box (vehicle) is inside the road area
    isVehicleInRoad(bbox) {
        return this.getVehicleLane(bbox) !== null;
    }

    // Calculate angle difference, handling wrap-around (-180 to 180)
    getAngleDifference(angle1, angle2) {
        let diff = angle2 - angle1;

        // Normalize to -180 to 180
        while (diff > 180) diff -= 360;
        while (diff < -180) diff += 360;

        return diff;
    }

    // Determine turn direction based on vehicle angle and current lane
    // Returns: 'left_turn', 'right_turn', or 'straight'
    getTurnDirection(vehicleAngle, currentLane) {
        if (!currentLane || !this.laneBaseAngles[currentLane]) {
            return 'straight';
        }

        const baseAngle = this.laneBaseAngles[currentLane];
        const angleDiff = this.getAngleDifference(baseAngle, vehicleAngle);

        // Left turn: -30 degrees from base
        // Right turn: +30 degrees from base
        if (angleDiff <= -this.turnAngleThreshold) {
            return 'left_turn';
        } else if (angleDiff >= this.turnAngleThreshold) {
            return 'right_turn';
        } else {
            return 'straight';
        }
    }

    // Determine if vehicle should change lanes based on position and direction
    // Returns new lane name or null if no change
    determineLaneChange(currentLane, vehicleAngle, bbox) {
        if (!currentLane) return null;

        const turnDirection = this.getTurnDirection(vehicleAngle, currentLane);

        // Only straight - no lane change
        if (turnDirection === 'straight') {
            return null;
        }

        // Get target lane based on turn direction
        const rules = this.laneChangeRules[currentLane];
        if (!rules) return null;

        const targetLane = rules[turnDirection];
        if (!targetLane) return null;

        // Check if vehicle is in the overlapping area (can be detected in target lane)
        const isInTargetLane = this.getVehicleLane(bbox) === targetLane;

        // Only change lanes if vehicle is actually in the target lane area
        if (isInTargetLane) {
            return targetLane;
        }

        return null;
    }
    // Load ONNX model
    async loadModel() {
        try {
            console.log('='.repeat(50));
            console.log('Loading traffic detection model...');
            console.log('='.repeat(50));

            // Use ONNX Runtime Web with WebGL acceleration (fallback to WASM)
            // WebGL is significantly faster than WASM for large models
            this.model = await ort.InferenceSession.create('./mvt_rtdetr.onnx', {
                executionProviders: ['webgl', 'wasm'],
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
            const inputArray = new Float32Array(1 * 3 * 640 * 640);

            for (let i = 0; i < 640 * 640; i++) {
                inputArray[i] = pixels[i * 4] / 255.0;
                inputArray[640 * 640 + i] = pixels[i * 4 + 1] / 255.0;
                inputArray[640 * 640 * 2 + i] = pixels[i * 4 + 2] / 255.0;
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

                // Class name mapping
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

                // Check which lane the vehicle is in
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

            // Filter to only keep vehicles inside the road
            const detections_in_road = detections.filter(d => d.in_road);

            // Count vehicles per lane
            const laneCounts = { AREA1: 0, AREA2: 0, AREA3: 0, AREA4: 0 };
            detections_in_road.forEach(d => {
                if (d.lane) laneCounts[d.lane]++;
            });

            console.log(`Detected ${detections.length} objects (${detections_in_road.length} in road)`);
            console.log(`Lane distribution: AREA1=${laneCounts.AREA1}, AREA2=${laneCounts.AREA2}, AREA3=${laneCounts.AREA3}, AREA4=${laneCounts.AREA4}`);
            return detections_in_road;

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

    // Draw road lane areas with semi-transparent fill
    drawRoadEdges(ctx) {
        try {
            // Draw each lane area with semi-transparent fill
            for (const lane of this.lanePolygons) {
                ctx.fillStyle = lane.color;
                ctx.beginPath();

                // Start from first point
                if (lane.polygon.length > 0) {
                    ctx.moveTo(lane.polygon[0][0], lane.polygon[0][1]);

                    // Draw to all other points
                    for (let i = 1; i < lane.polygon.length; i++) {
                        ctx.lineTo(lane.polygon[i][0], lane.polygon[i][1]);
                    }

                    // Close the path
                    ctx.closePath();
                    ctx.fill();
                }
            }

        } catch (error) {
            console.error('Drawing lane areas error:', error);
        }
    }

    // Update statistics
    updateStats(detections) {
        this.trafficStats.total_detections = detections.length;
        this.trafficStats.car_count = detections.filter(d => d.class_name === 'car').length;
        this.trafficStats.motorcycle_count = detections.filter(d => d.class_name === 'motorcycle').length;
    }
}

// ================= Simple IoU Tracker with Lane Change Detection =================
class SimpleTracker {
    constructor(detector) {
        this.detector = detector; // Reference to TrafficDetector for lane detection
        this.tracks = [];
        this.nextId = 1;
        this.maxAge = 5; // Maximum frames to keep track without detection
    }

    // Calculate IoU (Intersection over Union)
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

    // Get bottom center point of bbox (where vehicle touches road)
    getBottomCenter(bbox) {
        const [x1, y1, x2, y2] = bbox;
        return { x: (x1 + x2) / 2, y: y2 };
    }

    // Calculate angle from two points
    calculateAngle(p1, p2) {
        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const radians = Math.atan2(dy, dx);
        return radians * (180 / Math.PI);
    }

    // Update tracks with new detections
    update(detections, timestamp = Date.now()) {
        const matched = [];

        detections.forEach(det => {
            let bestMatch = null;
            let bestIoU = 0.3; // IoU threshold

            this.tracks.forEach(track => {
                const iou = this.calculateIoU(det.bbox, track.bbox);
                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestMatch = track;
                }
            });

            if (bestMatch) {
                // Update existing track
                const oldCenter = this.getBottomCenter(bestMatch.bbox);
                const newCenter = this.getBottomCenter(det.bbox);

                // Calculate velocity and angle
                const dx = newCenter.x - oldCenter.x;
                const dy = newCenter.y - oldCenter.y;
                const timeDelta = (timestamp - bestMatch.timestamp) / 1000; // seconds

                const velocityX = timeDelta > 0 ? dx / timeDelta : 0;
                const velocityY = timeDelta > 0 ? dy / timeDelta : 0;
                const speed = Math.sqrt(velocityX * velocityX + velocityY * velocityY);

                // Calculate movement angle
                const angle = this.calculateAngle(oldCenter, newCenter);

                // Determine lane change based on angle and position
                const currentLane = det.lane;
                const turnDirection = this.detector.getTurnDirection(angle, currentLane);
                const newLane = this.detector.determineLaneChange(currentLane, angle, det.bbox);

                // Track lane change event
                let laneChangeEvent = null;
                if (newLane && newLane !== bestMatch.currentLane) {
                    laneChangeEvent = {
                        from: bestMatch.currentLane,
                        to: newLane,
                        turnDirection: turnDirection,
                        timestamp: timestamp
                    };
                    console.log(`🔄 Vehicle #${bestMatch.id} lane change: ${bestMatch.currentLane} → ${newLane} (${turnDirection})`);
                }

                // Update track
                bestMatch.bbox = det.bbox;
                bestMatch.age = 0;
                bestMatch.timestamp = timestamp;
                bestMatch.velocity = { x: velocityX, y: velocityY };
                bestMatch.speed = speed;
                bestMatch.angle = angle;
                bestMatch.currentLane = newLane || currentLane;
                bestMatch.turnDirection = turnDirection;
                bestMatch.history.push(newCenter);

                // Keep only last 10 positions
                if (bestMatch.history.length > 10) {
                    bestMatch.history.shift();
                }

                // Record lane change
                if (laneChangeEvent) {
                    bestMatch.laneChanges.push(laneChangeEvent);
                }

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
                // New track
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
                    laneChanges: [] // Track all lane changes
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

        // Remove old tracks
        this.tracks = this.tracks.filter(track => {
            track.age++;
            return track.age < this.maxAge;
        });

        return matched;
    }
}

// ================= Global Variables =================
let detector = null;
let tracker = null; // Add tracker (kept for compatibility, but not used in main thread)
let detectionWorker = null; // Web Worker for detection
let isDetecting = false;
let detectionLoopId = null; // For requestAnimationFrame
let lastDetectionTime = 0; // Timestamp of last detection
const DETECTION_INTERVAL = 3000; // ms between detections
let isFirstDetection = true;  // Track if this is the first detection
let sseEventSource = null; // SSE connection for streaming frames
let latestFrameImage = null; // Latest frame from SSE stream
let workerReady = false; // Track if worker is initialized

// ================= Real-time Clock (GMT+8) =================
function updateClock() {
    // Get current time in GMT+8 (Taipei timezone)
    const now = new Date();
    const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
    const gmt8Time = new Date(utc + (3600000 * 8));

    // Format time: HH:MM:SS
    const hours = String(gmt8Time.getHours()).padStart(2, '0');
    const minutes = String(gmt8Time.getMinutes()).padStart(2, '0');
    const seconds = String(gmt8Time.getSeconds()).padStart(2, '0');
    const timeString = `${hours}:${minutes}:${seconds}`;

    // Format date: YYYY年MM月DD日
    const year = gmt8Time.getFullYear();
    const month = String(gmt8Time.getMonth() + 1).padStart(2, '0');
    const day = String(gmt8Time.getDate()).padStart(2, '0');
    const weekdays = ['星期日', '星期一', '星期二', '星期三', '星期四', '星期五', '星期六'];
    const weekday = weekdays[gmt8Time.getDay()];
    const dateString = `${year} 年 ${month} 月 ${day} 日 ${weekday}`;

    // Update main page time
    const currentTimeEl = document.getElementById('currentTime');
    const currentDateEl = document.getElementById('currentDate');
    if (currentTimeEl) currentTimeEl.textContent = timeString;
    if (currentDateEl) currentDateEl.textContent = dateString;

    // Update backend page time
    const backendTimeEl = document.getElementById('backendCurrentTime');
    const backendDateEl = document.getElementById('backendCurrentDate');
    if (backendTimeEl) backendTimeEl.textContent = timeString;
    if (backendDateEl) backendDateEl.textContent = dateString;
}

// ================= Initialization =================
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Application initializing...');

    // Initialize Web Worker for detection
    console.log('🧵 Starting detection worker thread...');
    detectionWorker = new Worker('./backends/detection-worker.js');

    // Setup worker message handler
    detectionWorker.onmessage = handleWorkerMessage;
    detectionWorker.onerror = (error) => {
        console.error('❌ Worker error:', error);
        alert('Detection worker failed to start');
    };

    // Initialize worker
    detectionWorker.postMessage({ type: 'init' });

    // Keep detector instance for drawing (UI thread only)
    detector = new TrafficDetector();
    console.log('✅ Main thread TrafficDetector created for UI rendering');

    // Setup event listeners
    setupEventListeners();

    // Start real-time clock (update every second)
    updateClock(); // Initial update
    setInterval(updateClock, 1000); // Update every 1 second

    console.log('Application initialized successfully');
});

// ================= Worker Message Handler =================
function handleWorkerMessage(e) {
    const { type, success, annotatedImageData, detections, laneStats, trafficFlow, statisticsDuration, inferenceTime, error } = e.data;

    switch (type) {
        case 'init_complete':
            if (success) {
                console.log('✅ Detection worker initialized successfully');
                workerReady = true;
            } else {
                console.error('❌ Worker initialization failed:', error);
                const errorMsg = `Detection worker initialization failed: ${error || 'Unknown error'}`;
                alert(errorMsg + '\n\nPlease check the browser console for details.');
                workerReady = false;
            }
            break;

        case 'detection_result':
            if (success) {
                console.log(`📊 [Main] Received detection results: ${detections.length} objects`);
                console.log('[Main] Displaying pre-rendered image from worker...');

                // Worker has already drawn everything - just display it!
                displayAnnotatedImage(annotatedImageData, laneStats, detections);

                // Update traffic flow statistics
                if (trafficFlow && statisticsDuration !== undefined) {
                    updateTrafficFlowDisplay(trafficFlow, statisticsDuration);
                }

                // Update statistics to server
                updateStatsToServer(detections, inferenceTime);

                // Hide loading animation
                const loading = document.getElementById('loading');
                if (loading && isFirstDetection) {
                    loading.style.display = 'none';
                    isFirstDetection = false;
                }
            } else {
                console.error('❌ Detection failed in worker:', error);
                // Don't alert for every detection failure in continuous mode
                if (!isDetecting) {
                    alert(`Detection failed: ${error}`);
                }
            }
            break;

        default:
            console.warn('Unknown message type from worker:', type);
    }
}

// ================= Display Annotated Image from Worker =================
function displayAnnotatedImage(imageData, laneStats, detections) {
    try {
        // Convert ImageData to Canvas
        const canvas = document.createElement('canvas');
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        const ctx = canvas.getContext('2d');
        ctx.putImageData(imageData, 0, 0);

        // Display the canvas
        const detectionImage = document.getElementById('detectionImage');
        const noImagePlaceholder = document.getElementById('noImagePlaceholder');

        if (detectionImage && noImagePlaceholder) {
            // Convert canvas to blob for display
            canvas.toBlob((blob) => {
                if (blob) {
                    const url = URL.createObjectURL(blob);
                    // Revoke old URL to prevent memory leak
                    if (detectionImage.src && detectionImage.src.startsWith('blob:')) {
                        URL.revokeObjectURL(detectionImage.src);
                    }
                    detectionImage.src = url;
                    detectionImage.style.display = 'block';
                    noImagePlaceholder.style.display = 'none';
                }
            }, 'image/jpeg', 0.75);
        }

        // Update lane statistics in UI (batch updates with requestAnimationFrame)
        requestAnimationFrame(() => {
            const updates = [
                ['area1Cars', laneStats.AREA1.cars],
                ['area1Motorcycles', laneStats.AREA1.motorcycles],
                ['area1Total', laneStats.AREA1.total],
                ['area2Cars', laneStats.AREA2.cars],
                ['area2Motorcycles', laneStats.AREA2.motorcycles],
                ['area2Total', laneStats.AREA2.total],
                ['area3Cars', laneStats.AREA3.cars],
                ['area3Motorcycles', laneStats.AREA3.motorcycles],
                ['area3Total', laneStats.AREA3.total],
                ['area4Cars', laneStats.AREA4.cars],
                ['area4Motorcycles', laneStats.AREA4.motorcycles],
                ['area4Total', laneStats.AREA4.total]
            ];

            updates.forEach(([id, value]) => {
                const element = document.getElementById(id);
                if (element) element.textContent = value;
            });

            // Update backend data page
            updateBackendDataPage(laneStats);
        });

        console.log('[Main] ✅ Image and statistics displayed');

    } catch (error) {
        console.error('[Main] Error displaying annotated image:', error);
    }
}

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

// ================= SSE Stream Connection =================
function startSSEStream() {
    if (sseEventSource) {
        console.log('SSE stream already connected');
        return;
    }

    console.log('📡 Connecting to SSE stream...');
    sseEventSource = new EventSource('/stream_frames');

    sseEventSource.onmessage = async (event) => {
        try {
            const data = JSON.parse(event.data);
            latestFrameImage = await loadImage(data.image);
            console.log(`📥 Received frame from SSE stream (${data.timestamp})`);
        } catch (err) {
            console.error('Error processing SSE frame:', err);
        }
    };

    sseEventSource.onerror = (error) => {
        console.error('❌ SSE connection error:', error);
        sseEventSource.close();
        sseEventSource = null;
        latestFrameImage = null;

        // Retry after 3 seconds
        setTimeout(() => {
            if (isDetecting) {
                console.log('🔄 Retrying SSE connection...');
                startSSEStream();
            }
        }, 3000);
    };

    sseEventSource.onopen = () => {
        console.log('✅ SSE stream connected');
    };
}

function stopSSEStream() {
    if (sseEventSource) {
        console.log('⏹️  Closing SSE stream...');
        sseEventSource.close();
        sseEventSource = null;
        latestFrameImage = null;
    }
}

// ================= Start/Stop Detection =================
async function toggleDetection() {
    const detectBtn = document.getElementById('detectBtn');

    if (!workerReady) {
        alert('Detection worker is not ready yet, please wait...');
        return;
    }

    if (!isDetecting) {
        // Start detection
        isDetecting = true;
        detectBtn.textContent = '⏸️ Stop Detection';
        detectBtn.style.background = 'linear-gradient(135deg, #f44336 0%, #e91e63 100%)';

        console.log('Starting real-time detection with SSE stream');

        // Start SSE stream
        startSSEStream();

        // Wait a bit for first frame
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Execute immediately once
        await performDetection();
        lastDetectionTime = performance.now();

        // Start requestAnimationFrame loop
        function detectionLoop(timestamp) {
            // Check if enough time has passed since last detection
            if (timestamp - lastDetectionTime >= DETECTION_INTERVAL) {
                // Perform detection and update timestamp after completion
                performDetection().then(() => {
                    lastDetectionTime = timestamp;
                }).catch(err => {
                    console.error('Detection error in loop:', err);
                    lastDetectionTime = timestamp; // Still update to prevent stuck state
                });
            }

            // Continue loop if still detecting
            if (isDetecting) {
                detectionLoopId = requestAnimationFrame(detectionLoop);
            }
        }

        // Start the loop
        detectionLoopId = requestAnimationFrame(detectionLoop);

    } else {
        // Stop detection
        isDetecting = false;
        detectBtn.textContent = '🔍 Start Detection';
        detectBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';

        // Cancel requestAnimationFrame
        if (detectionLoopId) {
            cancelAnimationFrame(detectionLoopId);
            detectionLoopId = null;
        }

        // Stop SSE stream
        stopSSEStream();

        // Reset first detection flag for next time
        isFirstDetection = true;

        console.log('Detection stopped');
    }
}

// ================= Perform Detection =================
async function performDetection() {
    if (!workerReady) {
        console.error('[Main] Worker not ready');
        return;
    }

    try {
        console.log('[Main] Processing frame for detection...');

        // Show loading animation only on first detection
        const loading = document.getElementById('loading');
        if (loading && isFirstDetection) {
            loading.style.display = 'block';
        }

        // Use cached frame from SSE stream if available, otherwise fetch
        let image;
        if (latestFrameImage) {
            image = latestFrameImage;
            console.log('[Main] ✅ Using cached frame from SSE stream');
        } else {
            console.log('[Main] ⚠️  No SSE frame available, falling back to fetch');
            const response = await fetch('/get_latest_frame');
            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'No frame available');
            }

            image = await loadImage(data.image);
        }

        console.log(`[Main] Image ready: ${image.width}x${image.height}`);

        // Convert image to ImageData for worker
        const canvas = document.createElement('canvas');
        canvas.width = 640;
        canvas.height = 640;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(image, 0, 0, 640, 640);
        const imageData = ctx.getImageData(0, 0, 640, 640);

        // Send detection task to worker (non-blocking)
        // Worker will handle: detection + tracking + drawing
        console.log('[Main] 🚀 Sending frame to worker (detection + rendering)...');
        detectionWorker.postMessage({
            type: 'detect',
            data: { imageData }
        });

        // Main thread continues immediately without blocking!
        console.log('[Main] ✅ Main thread free - worker processing in background');

    } catch (error) {
        console.error('[Main] Detection setup failed:', error);

        const loading = document.getElementById('loading');
        if (loading) loading.style.display = 'none';

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
    // ===== OPTIMIZATION 4: Use toBlob() with lower quality (async, non-blocking) =====
    const detectionImage = document.getElementById('detectionImage');
    const noImagePlaceholder = document.getElementById('noImagePlaceholder');

    if (detectionImage && noImagePlaceholder) {
        // Use toBlob with quality 0.75 (25% less data, much faster)
        canvas.toBlob((blob) => {
            if (blob) {
                const url = URL.createObjectURL(blob);
                // Revoke old URL to prevent memory leak
                if (detectionImage.src && detectionImage.src.startsWith('blob:')) {
                    URL.revokeObjectURL(detectionImage.src);
                }
                detectionImage.src = url;
                detectionImage.style.display = 'block';
                noImagePlaceholder.style.display = 'none';
            }
        }, 'image/jpeg', 0.75); // Quality 0.75 (was 0.9)
    }

    // Calculate lane-based statistics
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

    // ===== OPTIMIZATION 7: Batch DOM updates with requestAnimationFrame =====
    // Collect all DOM updates and apply in one frame to minimize reflow
    requestAnimationFrame(() => {
        // Update main page lane statistics (batch all reads/writes)
        const updates = [
            ['area1Cars', laneStats.AREA1.cars],
            ['area1Motorcycles', laneStats.AREA1.motorcycles],
            ['area1Total', laneStats.AREA1.total],
            ['area2Cars', laneStats.AREA2.cars],
            ['area2Motorcycles', laneStats.AREA2.motorcycles],
            ['area2Total', laneStats.AREA2.total],
            ['area3Cars', laneStats.AREA3.cars],
            ['area3Motorcycles', laneStats.AREA3.motorcycles],
            ['area3Total', laneStats.AREA3.total],
            ['area4Cars', laneStats.AREA4.cars],
            ['area4Motorcycles', laneStats.AREA4.motorcycles],
            ['area4Total', laneStats.AREA4.total]
        ];

        // Apply all updates at once
        updates.forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) element.textContent = value;
        });

        // Update backend data page in same frame
        updateBackendDataPage(laneStats);
    });
}

// ================= Update Traffic Flow Display =================
function updateTrafficFlowDisplay(trafficFlow, statisticsDuration) {
    // Format duration as HH:MM:SS
    const hours = Math.floor(statisticsDuration / 3600);
    const minutes = Math.floor((statisticsDuration % 3600) / 60);
    const seconds = statisticsDuration % 60;
    const durationStr = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;

    // Update each lane's traffic flow statistics
    ['AREA1', 'AREA2', 'AREA3', 'AREA4'].forEach((area, idx) => {
        const data = trafficFlow[area];
        if (!data) return;

        const laneNum = idx + 1;

        // Update cumulative count
        const totalElement = document.getElementById(`area${laneNum}TotalFlow`);
        if (totalElement) totalElement.textContent = data.totalCount;

        const carsElement = document.getElementById(`area${laneNum}CarsFlow`);
        if (carsElement) carsElement.textContent = data.totalCars;

        const motorcyclesElement = document.getElementById(`area${laneNum}MotorcyclesFlow`);
        if (motorcyclesElement) motorcyclesElement.textContent = data.totalMotorcycles;

        // Update duration
        const durationElement = document.getElementById(`area${laneNum}Duration`);
        if (durationElement) durationElement.textContent = durationStr;

        // Backend page
        const backendTotalElement = document.getElementById(`backendArea${laneNum}TotalFlow`);
        if (backendTotalElement) backendTotalElement.textContent = data.totalCount;

        const backendCarsElement = document.getElementById(`backendArea${laneNum}CarsFlow`);
        if (backendCarsElement) backendCarsElement.textContent = data.totalCars;

        const backendMotorcyclesElement = document.getElementById(`backendArea${laneNum}MotorcyclesFlow`);
        if (backendMotorcyclesElement) backendMotorcyclesElement.textContent = data.totalMotorcycles;

        const backendDurationElement = document.getElementById(`backendArea${laneNum}Duration`);
        if (backendDurationElement) backendDurationElement.textContent = durationStr;
    });
}

// ================= Update Backend Data Page =================
function updateBackendDataPage(laneStats) {
    // ===== OPTIMIZATION 7: Batch backend page updates =====
    const backendUpdates = [
        ['backendArea1Cars', laneStats.AREA1.cars],
        ['backendArea1Motorcycles', laneStats.AREA1.motorcycles],
        ['backendArea1Total', laneStats.AREA1.total],
        ['backendArea2Cars', laneStats.AREA2.cars],
        ['backendArea2Motorcycles', laneStats.AREA2.motorcycles],
        ['backendArea2Total', laneStats.AREA2.total],
        ['backendArea3Cars', laneStats.AREA3.cars],
        ['backendArea3Motorcycles', laneStats.AREA3.motorcycles],
        ['backendArea3Total', laneStats.AREA3.total],
        ['backendArea4Cars', laneStats.AREA4.cars],
        ['backendArea4Motorcycles', laneStats.AREA4.motorcycles],
        ['backendArea4Total', laneStats.AREA4.total]
    ];

    // Apply all updates at once
    backendUpdates.forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) element.textContent = value;
    });

    // Note: Track IDs are still maintained in detections data for internal use
    // but are not displayed in the UI
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
