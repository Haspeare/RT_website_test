# RT_website_test
A website for traffic flow detection. (testing)

## 檔案結構
```
RT_website_test/
├── backends/
│   ├── app.js          # 前端 JavaScript（偵測邏輯）
│   └── server.js       # Node.js 後端伺服器
├── libs/               # 模型處理套件（web）
├── node_modules/       # node模組
├── pkgs/
│   ├── package.json
│   └── package-lock.json
├── tools/
│   ├── test-worker.html    # 測試worker是否正確載入模型檔案（.onnx）
│   └── road_edge_marker.html # 標記路線邊緣或其他線條
└── index.html          # 主頁面
```
