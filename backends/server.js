// server.js - Traffic Detection System Node.js Server
// Integrated with record_hls.js functionality

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { spawn, exec } = require('child_process');

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '..')));  // Serve from project root

// HLS Stream URL (from record_hls.js)
const M3U8_URL = "https://jtmctrafficcctv2.gov.taipei/NVR/f0a5bf25-956f-4343-bed3-59df341071ea/live.m3u8";

// ================= Route Definitions =================

// Serve index.html
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'index.html'));
});

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        model_loaded: fs.existsSync(path.join(__dirname, '..', 'vmt_rtdetr.onnx')),
        model_type: 'onnx',
        stream_url: M3U8_URL
    });
});

// Get HLS stream info
app.get('/stream_info', (req, res) => {
    res.json({
        url: M3U8_URL,
        type: 'HLS',
        description: 'Taipei City Traffic Camera'
    });
});

// Get stream frame rate (from record_hls.js)
function getStreamFrameRate(callback) {
    const isWindows = process.platform === 'win32';
    const grepCmd = isWindows ? 'findstr /R "tbr"' : 'grep -oP "\\d+\\.?\\d* tbr"';

    exec(`ffmpeg -i ${M3U8_URL} 2>&1 | ${grepCmd}`, (error, stdout, stderr) => {
        if (error) {
            console.error(`Failed to get frame rate: ${error.message}`);
            callback(11); // Default value
            return;
        }
        const output = stdout || stderr;
        const match = output.match(/(\d+\.?\d*)/);
        const frameRate = match ? parseFloat(match[1]) : 11;
        callback(frameRate);
    });
}

// Capture single frame from HLS stream (using record_hls.js method)
app.get('/capture_frame', async (req, res) => {
    const outputPath = path.join(__dirname, '..', 'stream_snapshot.jpg');

    const requestStartTime = Date.now();
    console.log('📸 Capturing frame from HLS stream...');

    let responseSent = false;
    let ffmpegStartTime = Date.now();

    // Use ffmpeg with ultra-low latency settings for 0.5s target
    // Resize to 640x640 for faster processing and consistent model input
    const ffmpeg = spawn('ffmpeg', [
        '-y',  // Overwrite existing file
        '-hide_banner',
        '-loglevel', 'panic',  // Suppress all output for maximum speed
        '-analyzeduration', '100000',  // Ultra-fast analysis: 0.1 second
        '-probesize', '32768',  // Minimal probe size: 32KB
        '-max_delay', '0',  // No delay
        '-fflags', 'nobuffer+fastseek+flush_packets',  // Maximum speed flags
        '-flags', 'low_delay',  // Low delay mode
        '-strict', 'experimental',
        '-reconnect', '1',
        '-reconnect_streamed', '1',
        '-reconnect_delay_max', '0',  // No reconnect delay
        '-i', M3U8_URL,
        '-vf', 'scale=640:640:flags=fast_bilinear',  // Fast scaling
        '-frames:v', '1',  // Capture only one frame
        '-c:v', 'mjpeg',  // Use MJPEG for faster encoding
        '-f', 'image2',
        '-preset', 'ultrafast',  // Fastest preset
        '-tune', 'zerolatency',  // Zero latency tuning
        outputPath
    ]);

    let errorOutput = '';

    ffmpeg.stderr.on('data', (data) => {
        errorOutput += data.toString();
    });

    ffmpeg.on('close', (code) => {
        if (responseSent) return;
        responseSent = true;

        const ffmpegEndTime = Date.now();
        const captureTime = ffmpegEndTime - ffmpegStartTime;

        if (code === 0 && fs.existsSync(outputPath)) {
            const processingStartTime = Date.now();

            try {
                // Read image and convert to base64
                const imageBuffer = fs.readFileSync(outputPath);
                const base64Image = imageBuffer.toString('base64');

                const processingEndTime = Date.now();
                const processingTime = processingEndTime - processingStartTime;
                const totalTime = processingEndTime - requestStartTime;

                console.log('✅ Stream capture successful');
                console.log(`   ⏱️  Capture time: ${captureTime}ms`);
                console.log(`   ⏱️  Processing time (read + base64): ${processingTime}ms`);
                console.log(`   ⏱️  Total time: ${totalTime}ms`);

                res.json({
                    success: true,
                    image: `data:image/jpeg;base64,${base64Image}`,
                    timestamp: Date.now(),
                    source: 'HLS Stream (m3u8)',
                    timing: {
                        captureTime,
                        processingTime,
                        totalTime
                    }
                });
            } catch (readError) {
                console.error('❌ Error reading image:', readError);
                res.status(500).json({
                    success: false,
                    error: 'Failed to read captured image',
                    details: readError.message
                });
            }
        } else {
            console.error('❌ Stream capture failed:', errorOutput.substring(0, 200));
            console.log(`   ⏱️  Failed after: ${captureTime}ms`);
            res.status(500).json({
                success: false,
                error: 'Stream capture failed',
                details: errorOutput.substring(0, 500)
            });
        }
    });

    ffmpeg.on('error', (err) => {
        if (responseSent) return;
        responseSent = true;

        console.error('❌ FFmpeg execution error:', err);

        if (err.code === 'ENOENT') {
            res.status(500).json({
                success: false,
                error: 'FFmpeg not installed',
                details: 'Please install FFmpeg: sudo apt install ffmpeg'
            });
        } else {
            res.status(500).json({
                success: false,
                error: 'FFmpeg execution failed',
                details: err.message
            });
        }
    });
});

// Statistics data
let detectionStats = {
    total_detections: 0,
    car_count: 0,
    motorcycle_count: 0,
    last_update: null
};

// Update statistics
app.post('/update_stats', (req, res) => {
    const { detections, inferenceTime } = req.body;

    if (!detections || !Array.isArray(detections)) {
        return res.status(400).json({ error: 'Invalid detection data' });
    }

    detectionStats.total_detections = detections.length;
    detectionStats.car_count = detections.filter(d => d.class_name === 'car').length;
    detectionStats.motorcycle_count = detections.filter(d => d.class_name === 'motorcycle').length;
    detectionStats.last_update = new Date().toISOString();

    // Log inference time to terminal
    if (inferenceTime !== undefined) {
        console.log(`   🧠 Model inference time: ${inferenceTime}ms`);
    }

    res.json({ success: true, stats: detectionStats });
});

// Get statistics
app.get('/stats', (req, res) => {
    res.json({
        traffic_stats: detectionStats,
        model_type: 'onnx',
        stream_url: M3U8_URL
    });
});

// Start stream recording (from record_hls.js)
let recordingProcess = null;
let isRecording = false;

app.post('/start_stream_recording', (req, res) => {
    if (isRecording) {
        return res.json({
            success: false,
            message: 'Recording already in progress'
        });
    }

    const { duration = 120 } = req.body;  // Default 120 seconds

    getStreamFrameRate((frameRate) => {
        const startTime = new Date().toISOString();
        console.log('🎥 Starting stream recording');
        console.log(`Frame rate: ${frameRate} fps`);
        console.log(`Duration: ${duration} seconds`);

        const outputPath = path.join(__dirname, '..', 'stream_recording.mp4');

        recordingProcess = spawn('ffmpeg', [
            '-y',
            '-fflags', '+genpts',
            '-reconnect', '1',
            '-reconnect_streamed', '1',
            '-reconnect_delay_max', '5',
            '-i', M3U8_URL,
            '-vf', `select='gt(scene,0.0003)',setpts=N/(${frameRate}*TB)`,
            '-r', frameRate.toString(),
            '-t', duration.toString(),
            '-c:v', 'libx264',
            '-an',
            '-metadata', `creation_time=${startTime}`,
            outputPath
        ]);

        isRecording = true;

        recordingProcess.stderr.on('data', (data) => {
            // Only log important messages
            const msg = data.toString();
            if (msg.includes('frame=') || msg.includes('error') || msg.includes('Error')) {
                console.log(`FFmpeg: ${msg.substring(0, 100)}`);
            }
        });

        recordingProcess.on('close', (code) => {
            isRecording = false;
            const endTime = new Date().toISOString();

            if (code === 0) {
                console.log('✅ Recording completed');
                console.log(`Start time: ${startTime}`);
                console.log(`End time: ${endTime}`);
                const timeDiff = (new Date(endTime) - new Date(startTime)) / 1000;
                console.log(`Duration: ${timeDiff} seconds`);
            } else {
                console.error(`❌ FFmpeg error code: ${code}`);
            }
        });

        res.json({
            success: true,
            message: 'Stream recording started',
            duration: duration,
            frameRate: frameRate,
            startTime: startTime
        });
    });
});

app.post('/stop_stream_recording', (req, res) => {
    if (recordingProcess && isRecording) {
        recordingProcess.kill('SIGTERM');
        isRecording = false;
        res.json({ success: true, message: 'Stream recording stopped' });
    } else {
        res.json({ success: false, message: 'No recording in progress' });
    }
});

// ================= Start Server =================
app.listen(PORT, '0.0.0.0', () => {
    console.log('='.repeat(50));
    console.log('🚀 Traffic Detection System Started');
    console.log(`📡 Server running at: http://localhost:${PORT}`);
    console.log(`🎥 HLS Stream: ${M3U8_URL}`);
    console.log('='.repeat(50));
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n⏹️  Shutting down server...');
    if (recordingProcess && isRecording) {
        recordingProcess.kill('SIGTERM');
    }
    process.exit(0);
});
