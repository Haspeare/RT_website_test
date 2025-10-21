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

    // Check which lane area a vehicle is in (if any)
    getVehicleLane(bbox) {
        // bbox format: [x1, y1, x2, y2]
        const [x1, y1, x2, y2] = bbox;

        // Calculate center point and bottom-center of the bounding box
        const centerX = (x1 + x2) / 2;
        const centerY = (y1 + y2) / 2;
        const bottomCenterX = (x1 + x2) / 2;
        const bottomCenterY = y2;

        // Check each lane area
        for (const lane of this.lanePolygons) {
            if (this.isPointInPolygon(bottomCenterX, bottomCenterY, lane.polygon) ||
                this.isPointInPolygon(centerX, centerY, lane.polygon)) {
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

                // Check which lane the vehicle is in
                const lane = this.getVehicleLane([x1, y1, x2, y2]);
                const isInRoad = lane !== null;

                detections.push({
                    class_id: cls,
                    class_name: className,
                    confidence: conf,
                    bbox: [x1, y1, x2, y2],
                    in_road: isInRoad,
                    lane: lane  // Add lane information (AREA1, AREA2, AREA3, AREA4, or null)
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
let tracker = null; // Add tracker
let isDetecting = false;
let detectionInterval = null;
let isFirstDetection = true;  // Track if this is the first detection

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

    // Create detector instance
    detector = new TrafficDetector();

    // Load ONNX model
    const loaded = await detector.loadModel();

    if (!loaded) {
        alert('Model loading failed, please confirm model.onnx file exists');
    }

    // Create tracker instance (after detector is created)
    tracker = new SimpleTracker(detector);
    console.log('Tracker initialized with lane change detection');

    // Setup event listeners
    setupEventListeners();

    // Start real-time clock (update every second)
    updateClock(); // Initial update
    setInterval(updateClock, 1000); // Update every 1 second

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

        // Apply tracking with lane change detection
        const trackedDetections = tracker ? tracker.update(detections, Date.now()) : detections;

        console.log(`Tracked ${trackedDetections.length} objects with IDs`);

        // Draw detection results (using TrafficDetector)
        const annotatedCanvas = detector.drawDetections(image, trackedDetections);
        console.log('Drawing complete, canvas:', annotatedCanvas ? `${annotatedCanvas.width}x${annotatedCanvas.height}` : 'null');

        if (annotatedCanvas) {
            // Display results
            displayResults(annotatedCanvas, trackedDetections);

            // Update statistics
            detector.updateStats(trackedDetections);

            // Update statistics to server (include inference time)
            await updateStatsToServer(trackedDetections, detector.lastInferenceTime);
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

    // Update main page lane statistics
    document.getElementById('area1Cars').textContent = laneStats.AREA1.cars;
    document.getElementById('area1Motorcycles').textContent = laneStats.AREA1.motorcycles;
    document.getElementById('area1Total').textContent = laneStats.AREA1.total;

    document.getElementById('area2Cars').textContent = laneStats.AREA2.cars;
    document.getElementById('area2Motorcycles').textContent = laneStats.AREA2.motorcycles;
    document.getElementById('area2Total').textContent = laneStats.AREA2.total;

    document.getElementById('area3Cars').textContent = laneStats.AREA3.cars;
    document.getElementById('area3Motorcycles').textContent = laneStats.AREA3.motorcycles;
    document.getElementById('area3Total').textContent = laneStats.AREA3.total;

    document.getElementById('area4Cars').textContent = laneStats.AREA4.cars;
    document.getElementById('area4Motorcycles').textContent = laneStats.AREA4.motorcycles;
    document.getElementById('area4Total').textContent = laneStats.AREA4.total;

    // Update backend data page
    updateBackendDataPage(laneStats);
}

// ================= Update Backend Data Page =================
function updateBackendDataPage(laneStats) {
    // Update backend page lane statistics with safe element access
    const updateElement = (id, value) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    };

    // AREA1
    updateElement('backendArea1Cars', laneStats.AREA1.cars);
    updateElement('backendArea1Motorcycles', laneStats.AREA1.motorcycles);
    updateElement('backendArea1Total', laneStats.AREA1.total);

    // AREA2
    updateElement('backendArea2Cars', laneStats.AREA2.cars);
    updateElement('backendArea2Motorcycles', laneStats.AREA2.motorcycles);
    updateElement('backendArea2Total', laneStats.AREA2.total);

    // AREA3
    updateElement('backendArea3Cars', laneStats.AREA3.cars);
    updateElement('backendArea3Motorcycles', laneStats.AREA3.motorcycles);
    updateElement('backendArea3Total', laneStats.AREA3.total);

    // AREA4
    updateElement('backendArea4Cars', laneStats.AREA4.cars);
    updateElement('backendArea4Motorcycles', laneStats.AREA4.motorcycles);
    updateElement('backendArea4Total', laneStats.AREA4.total);
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
