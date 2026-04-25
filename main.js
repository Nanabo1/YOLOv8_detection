// 全局变量
let isDetecting = false;
let logUpdateInterval = null;
let resultsUpdateInterval = null;

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    // 启动日志更新
    startLogUpdates();

    // 如果是检测页面，启动结果更新
    if (window.location.pathname.includes('detect')) {
        startResultsUpdates();
    }

    // 如果是历史记录页面，加载数据
    if (window.location.pathname === '/history') {
        loadHistoryData();
    }
});

// 启动日志更新
function startLogUpdates() {
    logUpdateInterval = setInterval(updateLogs, 2000);
}

// 更新日志
function updateLogs() {
    fetch('/api/logs')
        .then(response => response.json())
        .then(data => {
            const logArea = document.getElementById('log-area');
            if (logArea) {
                logArea.innerHTML = data.logs.map(log =>
                    `<div class="log-entry">${log}</div>`
                ).join('');
                // 滚动到底部
                logArea.scrollTop = logArea.scrollHeight;
            }
        })
        .catch(error => console.error('获取日志失败:', error));
}

// 启动结果更新
function startResultsUpdates() {
    resultsUpdateInterval = setInterval(updateResults, 1000);
}

// 更新检测结果
function updateResults() {
    fetch('/api/detection_results')
        .then(response => response.json())
        .then(data => {
            // 更新统计信息
            if (data.stats) {
                updateStats(data.stats);
            } else if (data.count !== undefined) {
                updateCount(data.count);
            }
        })
        .catch(error => console.error('获取结果失败:', error));
}

// 更新统计信息（速度页面）
function updateStats(stats) {
    const statsArea = document.getElementById('stats-area');
    if (!statsArea) return;

    let html = '<div class="stats-grid">';

    for (const [vehicleId, vehicleStats] of Object.entries(stats)) {
        html += `
            <div class="stat-card">
                <h5>车辆 ${vehicleId}</h5>
                <div class="stat-value">${vehicleStats.current.toFixed(1)} km/h</div>
                <div>最高: ${vehicleStats.max.toFixed(1)} km/h</div>
                <div>最低: ${vehicleStats.min.toFixed(1)} km/h</div>
                <div>平均: ${vehicleStats.avg.toFixed(1)} km/h</div>
            </div>
        `;
    }

    html += '</div>';
    statsArea.innerHTML = html;
}

// 更新车辆数量（普通检测页面）
function updateCount(count) {
    const countArea = document.getElementById('vehicle-count');
    if (countArea) {
        countArea.innerHTML = `
            <div class="stat-card">
                <h5>当前检测到的车辆</h5>
                <div class="stat-value">${count}</div>
            </div>
        `;
    }
}

// 启动摄像头
function startCamera(type, source) {
    fetch('/api/start_camera', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            type: type,
            source: source
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('摄像头启动成功');
        } else {
            alert('摄像头启动失败: ' + data.message);
        }
    })
    .catch(error => {
        alert('请求失败: ' + error);
    });
}

// 停止摄像头
function stopCamera() {
    fetch('/api/stop_camera', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('摄像头已关闭');
            // 停止检测
            if (isDetecting) {
                stopDetection();
            }
        }
    });
}

// 开始检测
function startDetection() {
    fetch('/api/start_detection', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            isDetecting = true;
            document.getElementById('start-detection').disabled = true;
            document.getElementById('stop-detection').disabled = false;
        } else {
            alert('启动检测失败: ' + data.message);
        }
    });
}

// 停止检测
function stopDetection() {
    fetch('/api/stop_detection', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            isDetecting = false;
            document.getElementById('start-detection').disabled = false;
            document.getElementById('stop-detection').disabled = true;
        }
    });
}

// 上传文件
function uploadFile(input) {
    const file = input.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/upload_file', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('文件上传成功');
        } else {
            alert('上传失败: ' + data.message);
        }
    });
}

// 加载历史数据
function loadHistoryData() {
    // 加载车辆记录
    fetch('/api/vehicle_records')
        .then(response => response.json())
        .then(data => {
            updateRecordsTable(data);
        });

    // 加载速度统计
    fetch('/api/speed_stats')
        .then(response => response.json())
        .then(data => {
            updateStatsTable(data);
        });
}

// 更新记录表格
function updateRecordsTable(records) {
    const tableBody = document.getElementById('records-table-body');
    if (!tableBody) return;

    tableBody.innerHTML = records.map(record => `
        <tr>
            <td>${record.id}</td>
            <td>${record.vehicle_id}</td>
            <td>${record.speed ? record.speed.toFixed(1) + ' km/h' : '--'}</td>
            <td>${record.vehicle_type || '--'}</td>
            <td>${record.timestamp}</td>
        </tr>
    `).join('');
}

// 更新统计表格
function updateStatsTable(stats) {
    const tableBody = document.getElementById('stats-table-body');
    if (!tableBody) return;

    tableBody.innerHTML = stats.map(stat => `
        <tr>
            <td>${stat.id}</td>
            <td>${stat.vehicle_id}</td>
            <td>${stat.max_speed.toFixed(1)} km/h</td>
            <td>${stat.min_speed.toFixed(1)} km/h</td>
            <td>${stat.avg_speed.toFixed(1)} km/h</td>
            <td>${stat.timestamp}</td>
        </tr>
    `).join('');
}

// 页面卸载时清理
window.addEventListener('beforeunload', function() {
    if (logUpdateInterval) {
        clearInterval(logUpdateInterval);
    }
    if (resultsUpdateInterval) {
        clearInterval(resultsUpdateInterval);
    }
});