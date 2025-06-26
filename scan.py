import os
import time
import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import psutil
import pynvml
import subprocess
import numpy as np
from collections import defaultdict
import threading
from flask import Flask, render_template, send_from_directory, jsonify, request

# 配置参数
DATA_DIR = "gpu_monitor_data"
LOG_INTERVAL = 30  # 秒
AGGREGATION_INTERVAL = 120  # 聚合间隔(秒)
MAX_GAP_TO_MERGE = 240  # 合并时间段的最大间隔(秒)
FLASK_PORT = 5000  # Flask服务器端口

# 创建Flask应用
app = Flask(__name__)
app.config['DATA_DIR'] = DATA_DIR
app.config['LOG_INTERVAL'] = LOG_INTERVAL
app.config['AGGREGATION_INTERVAL'] = AGGREGATION_INTERVAL

# 容器进程缓存
container_process_cache = {}

os.makedirs(DATA_DIR, exist_ok=True)

# 初始化GPU监控
try:
    pynvml.nvmlInit()
    GPU_COUNT = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
except Exception as e:
    print(f"GPU初始化失败: {e}")
    GPU_COUNT = 0

# ===== 系统监控功能 =====

def run_command(cmd):
    """执行命令并返回输出"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:
        print(f"命令执行失败: {cmd}, 错误: {e}")
    return ""

def get_container_info(pid):
    """获取PID对应的容器信息"""
    if pid == 0:
        return "host", "host", "host"  # 内核进程
    
    try:
        # 使用cgroup方法获取容器信息
        cgroup_path = f"/proc/{pid}/cgroup"
        if os.path.exists(cgroup_path):
            with open(cgroup_path, 'r') as f:
                for line in f:
                    # Docker 容器（不同格式）
                    if ":/docker/" in line:  # 标准Docker格式
                        container_id = line.split("/docker/")[-1].split('.scope')[0].strip()
                        return get_container_details(container_id)
                    
                    elif line.endswith(':/docker\n'):  # 某些旧版本的格式
                        container_id = line.split(' ')[-1].split('/')[-1].split('.scope')[0].strip()
                        return get_container_details(container_id)
                    
                    # Kubernetes/Podman
                    elif ":/kubepods/" in line:  # Kubernetes格式
                        match = re.search(r'([a-f0-9]{64})', line)
                        if match:
                            container_id = match.group(1)
                            return get_container_details(container_id)
                    
                    # Containerd/CRI-O
                    elif ":/system.slice/containerd.service" in line:
                        match = re.search(r'([a-f0-9]{64})', line)
                        if match:
                            container_id = match.group(1)
                            return get_container_details(container_id)
        
        # 使用docker top方法作为备选
        return find_container_by_pid(pid)
    
    except Exception as e:
        print(f"获取容器信息时出错 (PID={pid}): {e}")
        return "host", "host", "host"

def get_container_details(container_id):
    """通过容器ID获取详细信息"""
    try:
        # 获取容器名称
        cmd = f"docker inspect --format '{{{{.Name}}}}' {container_id}"
        container_name = run_command(cmd).lstrip('/') or f"container_{container_id[:12]}"
        
        # 获取服务名称（适用于Docker Compose）
        cmd = f"docker inspect --format '{{{{index .Config.Labels \"com.docker.compose.service\"}}}}' {container_id}"
        service_name = run_command(cmd)
        
        # 尝试从名称推断服务名称
        if not service_name or service_name == "unknown_service":
            if '-' in container_name:
                service_name = container_name.split('-')[0]
            else:
                service_name = container_name
        
        return container_id[:12], container_name, service_name
    except Exception as e:
        print(f"获取容器详情时出错 (ID={container_id}): {e}")
        return container_id[:12], f"container_{container_id[:12]}", "unknown_service"

def find_container_by_pid(pid):
    """备选方法：遍历所有容器查找PID"""
    try:
        # 获取所有运行的容器
        cmd = "docker ps -q"
        containers_output = run_command(cmd)
        if not containers_output:
            return "host", "host", "host"
        
        # 检查每个容器
        for container_id in containers_output.split('\n'):
            if not container_id:
                continue
            
            # 检查容器是否包含该进程
            top_cmd = f"docker top {container_id}"
            top_output = run_command(top_cmd)
            
            if not top_output:
                continue
            
            # 在容器进程列表中查找PID
            for process_line in top_output.split('\n')[1:]:  # 跳过标题行
                columns = process_line.split()
                if len(columns) >= 2:
                    try:
                        process_pid = int(columns[1])
                        if process_pid == pid:
                            return get_container_details(container_id)
                    except ValueError:
                        continue
        
        return "host", "host", "host"
    except Exception as e:
        print(f"备选容器查找失败 (PID={pid}): {e}")
        return "host", "host", "host"

def get_gpu_processes():
    """获取所有GPU进程信息"""
    try:
        pynvml.nvmlInit()
        processes = []
        
        device_count = pynvml.nvmlDeviceGetCount()
        for gpu_index in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            
            # 获取GPU总内存信息
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mem = mem_info.total // (1024 * 1024)  # MB
            
            # 获取GPU利用率
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu if hasattr(util, 'gpu') else 0
            
            # 获取计算和图形进程
            compute_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            graphics_processes = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            
            for proc in compute_processes + graphics_processes:
                # 获取显存使用量 (MB)
                used_mem =  (proc.usedGpuMemory or 0)  // (1024 * 1024) if hasattr(proc, 'usedGpuMemory') else 0
                
                # 获取进程信息
                try:
                    process = psutil.Process(proc.pid)
                    cmd = ' '.join(process.cmdline())[:100]
                    create_time = process.create_time()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    cmd = ""
                    create_time = time.time()
                
                processes.append({
                    "pid": proc.pid,
                    "gpu_index": gpu_index,
                    "gpu_utilization": gpu_util,
                    "used_memory_mb": used_mem,
                    "total_memory_mb": total_mem,
                    "memory_percent": (used_mem / total_mem) * 100 if total_mem > 0 else 0,
                    "command": cmd,
                    "start_time": create_time
                })
        return processes
    except pynvml.NVMLError as e:
        print(f"获取GPU进程信息失败: {e}")
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

def save_raw_data(data):
    """保存原始数据到JSON文件（自动格式化换行）"""
    if not data:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(DATA_DIR, f"raw_{timestamp}.json")
    
    # 创建目录（如果不存在）
    os.makedirs(DATA_DIR, exist_ok=True)
    
    with open(filename, 'w') as f:
        # 添加indent=4实现格式化换行
        json.dump({
            "timestamp": time.time(),
            "data": data
        }, f, indent=4)  # 关键修改：添加indent参数

def aggregate_data():
    """聚合数据到CSV"""
    # 找到所有原始数据文件
    raw_files = [f for f in os.listdir(DATA_DIR) if f.startswith("raw_")]
    # 创建聚合数据结构
    container_stats = defaultdict(lambda: {
        "container_id": "",
        "service_name": "",
        "container_name": "",
        "first_seen": float('inf'),
        "last_seen": 0,
        "total_gpu_seconds": 0,
        "max_memory_used": 0,
        "total_gpu_util": 0,
        "samples_count": 0,
        "pids": set()
    })
    
    gpu_stats = defaultdict(lambda: {
        "gpu_index": -1,
        "first_seen": float('inf'),
        "last_seen": 0,
        "max_memory_used": 0,
        "peak_gpu_util": 0,
        "total_gpu_util": 0,
        "util_samples": 0,
        "memory_samples": 0,
        "total_memory_used": 0
    })
    
    # 容器时间线数据结构
    container_timeline = defaultdict(list)

    for file in raw_files:
        print(f"处理文件: {file}")
        path = os.path.join(DATA_DIR, file)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                timestamp = data["timestamp"]
                for proc in data["data"]:
                    # 获取容器信息
                    container_id, container_name, service_name = get_container_info(proc["pid"])

                    if container_id == "host":
                        continue
                    
                    # 创建容器键
                    container_key = (container_id, service_name, container_name)
                    
                    # 更新容器统计
                    container = container_stats[container_key]
                    if not container["container_id"]:
                        container.update({
                            "container_id": container_id,
                            "service_name": service_name,
                            "container_name": container_name
                        })
                    
                    container["first_seen"] = min(container["first_seen"], timestamp)
                    container["last_seen"] = max(container["last_seen"], timestamp)
                    container["total_gpu_seconds"] += LOG_INTERVAL
                    container["max_memory_used"] = max(
                        container["max_memory_used"],
                        proc["used_memory_mb"]
                    )
                    container["total_gpu_util"] += proc["gpu_utilization"]
                    container["samples_count"] += 1
                    container["pids"].add(proc["pid"])
                    
                    # 更新GPU统计
                    gpu_key = proc["gpu_index"]
                    gpu = gpu_stats[gpu_key]
                    if gpu["gpu_index"] == -1:
                        gpu["gpu_index"] = gpu_key
                    
                    gpu["first_seen"] = min(gpu["first_seen"], timestamp)
                    gpu["last_seen"] = max(gpu["last_seen"], timestamp)
                    gpu["max_memory_used"] = max(
                        gpu["max_memory_used"],
                        proc["used_memory_mb"]
                    )
                    gpu["peak_gpu_util"] = max(
                        gpu["peak_gpu_util"],
                        proc["gpu_utilization"]
                    )
                    gpu["total_gpu_util"] += proc["gpu_utilization"]
                    gpu["util_samples"] += 1
                    gpu["total_memory_used"] += proc["used_memory_mb"]
                    gpu["memory_samples"] += 1
                    # 为容器记录时间点
                    container_timeline[container_key].append(timestamp)

                container_process_cache.clear()

        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
    
    # 保存容器统计
    if container_stats:
        container_csv = os.path.join(DATA_DIR, "container_stats.csv")
        container_header = [
            "timestamp", "container_id", "service_name", "container_name", 
            "first_seen", "last_seen", "total_gpu_seconds", "max_memory_used", 
            "avg_gpu_util", "unique_pids", "samples_count"
        ]
        
        container_exists = os.path.isfile(container_csv)
        with open(container_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=container_header)
            if not container_exists:
                writer.writeheader()
                
            for key, stats in container_stats.items():
                # 计算平均值
                stats["avg_gpu_util"] = stats["total_gpu_util"] / stats["samples_count"] if stats["samples_count"] > 0 else 0
                stats["unique_pids"] = len(stats["pids"])
                
                # 准备写入的行
                row = {
                    "timestamp": time.time(),
                    "container_id": stats["container_id"],
                    "service_name": stats["service_name"],
                    "container_name": stats["container_name"],
                    "first_seen": stats["first_seen"],
                    "last_seen": stats["last_seen"],
                    "total_gpu_seconds": stats["total_gpu_seconds"],
                    "max_memory_used": stats["max_memory_used"],
                    "avg_gpu_util": stats["avg_gpu_util"],
                    "unique_pids": stats["unique_pids"],
                    "samples_count": stats["samples_count"]
                }
                writer.writerow(row)
    
    # 保存GPU统计
    if gpu_stats:
        gpu_csv = os.path.join(DATA_DIR, "gpu_stats.csv")
        gpu_header = [
            "timestamp", "gpu_index", "first_seen", "last_seen", 
            "max_memory_used", "peak_gpu_util", "avg_gpu_util",
            "avg_memory_used", "samples_count"
        ]
        
        gpu_exists = os.path.isfile(gpu_csv)
        with open(gpu_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=gpu_header)
            if not gpu_exists:
                writer.writeheader()
                
            for key, stats in gpu_stats.items():
                # 计算平均值
                stats["avg_gpu_util"] = stats["total_gpu_util"] / stats["util_samples"] if stats["util_samples"] > 0 else 0
                stats["avg_memory_used"] = stats["total_memory_used"] / stats["memory_samples"] if stats["memory_samples"] > 0 else 0
                
                # 准备写入的行
                row = {
                    "timestamp": time.time(),
                    "gpu_index": stats["gpu_index"],
                    "first_seen": stats["first_seen"],
                    "last_seen": stats["last_seen"],
                    "max_memory_used": stats["max_memory_used"],
                    "peak_gpu_util": stats["peak_gpu_util"],
                    "avg_gpu_util": stats["avg_gpu_util"],
                    "avg_memory_used": stats["avg_memory_used"],
                    "samples_count": stats["memory_samples"]
                }
                writer.writerow(row)

    # 保存时间段数据
    if container_timeline:
        timeline_csv = os.path.join(DATA_DIR, "container_timeline.csv")
        timeline_exists = os.path.isfile(timeline_csv)
        
        with open(timeline_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if not timeline_exists:
                writer.writerow(["container_id", "container_name", "service_name", "start_time", "end_time"])
                
            for container_key, timestamps in container_timeline.items():
                if not timestamps:
                    continue
                    
                container_id, container_name, service_name = container_key
                # 排序时间戳
                timestamps.sort()
                
                # 合并连续的时间点
                merged_periods = []
                current_start = timestamps[0]
                current_end = timestamps[0]
                
                for i in range(1, len(timestamps)):
                    # 如果时间间隔小于最大允许间隔，则扩展当前时间段
                    if timestamps[i] - current_end <= MAX_GAP_TO_MERGE:
                        current_end = timestamps[i]
                    else:
                        # 保存当前时间段并开始新的时间段
                        merged_periods.append((current_start, current_end))
                        current_start = timestamps[i]
                        current_end = timestamps[i]
                
                # 添加最后一个时间段
                merged_periods.append((current_start, current_end))
                
                # 写入合并后的时间段
                for start_time, end_time in merged_periods:
                    # 确保时间段至少有一定的长度（2倍日志间隔）
                    if end_time - start_time >= LOG_INTERVAL * 2:
                        writer.writerow([
                            container_id,
                            container_name,
                            service_name,
                            start_time,
                            end_time
                        ])
    # 删除已处理的原始文件
    for file in raw_files:
        try:
            os.remove(os.path.join(DATA_DIR, file))
        except Exception as e:
            print(f"删除文件 {file} 失败: {e}")

def generate_dashboard():
    """生成交互式仪表盘"""
    try:
        # 读取容器统计数据
        container_path = os.path.join(DATA_DIR, "container_stats.csv")
        gpu_path = os.path.join(DATA_DIR, "gpu_stats.csv")
        
        if not os.path.exists(container_path):
            print("没有找到容器统计数据")
            return
        
        # 读取数据
        container_df = pd.read_csv(container_path)
        if 'timestamp' in container_df:
            container_df['timestamp'] = pd.to_datetime(container_df['timestamp'], unit='s')
        if 'first_seen' in container_df:
            container_df['first_seen'] = pd.to_datetime(container_df['first_seen'], unit='s')
        if 'last_seen' in container_df:
            container_df['last_seen'] = pd.to_datetime(container_df['last_seen'], unit='s')
        
        # 获取最近1小时的数据
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_containers = container_df[container_df['last_seen'] > recent_cutoff]

        #
        timeline_path = os.path.join(DATA_DIR, "container_timeline.csv")
        container_timeline = {}
        if os.path.exists(timeline_path):
            timeline_df = pd.read_csv(timeline_path)
            if 'start_time' in timeline_df:
                timeline_df['start_time'] = pd.to_datetime(timeline_df['start_time'], unit='s')
            if 'end_time' in timeline_df:
                timeline_df['end_time'] = pd.to_datetime(timeline_df['end_time'], unit='s')
                
            # 按容器名称分组
            container_groups = timeline_df.groupby(['container_id', 'container_name', 'service_name'])
            
            for container_key, group in container_groups:
                container_id, container_name, service_name = container_key
                display_name = f"{service_name} ({container_name})" if service_name != container_name else container_name
                
                # 合并可能的时间段重叠
                periods = []
                sorted_group = group.sort_values('start_time')
                
                if not sorted_group.empty:
                    current_start = sorted_group.iloc[0]['start_time']
                    current_end = sorted_group.iloc[0]['end_time']
                    
                    for i in range(1, len(sorted_group)):
                        next_start = sorted_group.iloc[i]['start_time']
                        next_end = sorted_group.iloc[i]['end_time']
                        
                        # 检查时间段是否重叠或相邻
                        if next_start <= current_end + timedelta(seconds=MAX_GAP_TO_MERGE):
                            # 扩展当前时间段
                            current_end = max(current_end, next_end)
                        else:
                            # 保存当前时间段并开始新的
                            periods.append({
                                "start": current_start,
                                "end": current_end,
                                "duration": (current_end - current_start).total_seconds()
                            })
                            current_start = next_start
                            current_end = next_end
                    
                    # 添加最后一个时间段
                    periods.append({
                        "start": current_start,
                        "end": current_end,
                        "duration": (current_end - current_start).total_seconds()
                    })
                
                container_timeline[display_name] = periods

        # 创建交互式HTML仪表盘
        dashboard_path = os.path.join(DATA_DIR, "gpu_dashboard.html")
        with open(dashboard_path, 'w') as f:
            f.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>GPU Usage Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    .dashboard {
                        display: grid;
                        grid-template-columns: repeat(2, 1fr);
                        gap: 20px;
                        padding: 20px;
                    }
                    .chart-container {
                        background-color: #fff;
                        border-radius: 8px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                        padding: 15px;
                    }
                    h1, h2 {
                        color: #333;
                    }
                    .container-gpu-table {
                        width: 100%;
                        border-collapse: collapse;
                    }
                    .container-gpu-table th, .container-gpu-table td {
                        padding: 8px;
                        border: 1px solid #ddd;
                        text-align: left;
                    }
                    .container-gpu-table th {
                        background-color: #f2f2f2;
                    }
                    .gpu-index {
                        display: inline-block;
                        padding: 2px 6px;
                        margin: 1px;
                        background-color: #007BFF;
                        color: white;
                        border-radius: 4px;
                        font-size: 0.9em;
                    }
                </style>
            </head>
            <body>
                <h1>GPU Usage Dashboard</h1>
                <div class="dashboard">
            """)
            
            # 1. 当前活跃容器
            f.write('<div class="chart-container">\n')
            f.write('<h2>Current Active Containers</h2>\n')
            f.write('<div id="active-containers-table"></div>\n')
            f.write("""
            <script>
                // 从API获取当前容器数据
                fetch('/api/containers')
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success' && data.data.length > 0) {
                            // 按容器分组
                            const containers = {};
                            data.data.forEach(item => {
                                const key = `${item.container_id} (${item.container_name})`;
                                if (!containers[key]) {
                                    containers[key] = {
                                        id: item.container_id,
                                        name: item.container_name,
                                        service: item.service_name,
                                        gpus: [],
                                        memory: 0,
                                        gpu_util: 0,
                                        pids: []
                                    };
                                }
                                containers[key].gpus.push(item.gpu_index);
                                containers[key].memory += item.memory_usage_mb;
                                containers[key].gpu_util = Math.max(containers[key].gpu_util, item.gpu_utilization);
                                containers[key].pids.push({pid: item.pid, mem: item.memory_usage_mb});
                            });
                            
                            // 生成表格
                            let tableHtml = `
                                <table class="container-gpu-table">
                                    <tr>
                                        <th>Container ID</th>
                                        <th>Name</th>
                                        <th>Service</th>
                                        <th>GPUs Used</th>
                                        <th>Memory (MB)</th>
                                        <th>GPU Util (%)</th>
                                        <th>PIDs</th>
                                    </tr>
                            `;
                            
                            Object.values(containers).forEach(container => {
                                tableHtml += `
                                    <tr>
                                        <td>${container.id}</td>
                                        <td>${container.name}</td>
                                        <td>${container.service}</td>
                                        <td>
                                `;
                                
                                // 添加GPU标签
                                container.gpus.forEach(gpu => {
                                    tableHtml += `<span class="gpu-index">GPU ${gpu}</span>`;
                                });
                                
                                tableHtml += `
                                        </td>
                                        <td>${container.memory.toFixed(1)}</td>
                                        <td>${container.gpu_util.toFixed(1)}</td>
                                        <td>
                                `;
                                
                                // 添加PID和内存使用
                                container.pids.forEach(pidInfo => {
                                    tableHtml += `${pidInfo.pid} (${pidInfo.mem}MB)<br>`;
                                });
                                
                                tableHtml += `
                                        </td>
                                    </tr>
                                `;
                            });
                            
                            tableHtml += `</table>`;
                            document.getElementById('active-containers-table').innerHTML = tableHtml;
                        } else {
                            document.getElementById('active-containers-table').innerHTML = '<p>No active GPU containers found</p>';
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching container data:', error);
                        document.getElementById('active-containers-table').innerHTML = '<p>Error loading data</p>';
                    });
            </script>
            """)
            f.write('</div>\n')
            
            # 2. 服务显存使用热力图
            f.write('<div class="chart-container">\n')
            f.write('<h2>Memory Usage by Service</h2>\n')
            if not container_df.empty and 'service_name' in container_df:
                service_memory = container_df.groupby('service_name')['max_memory_used'].sum().sort_values(ascending=False).head(10)
                if not service_memory.empty:
                    service_memory = service_memory.reset_index()
                    f.write('<div id="service-memory-chart"></div>\n')
                    f.write("""
                    <script>
                        var data = [{
                            type: 'bar',
                            x: %s,
                            y: %s,
                            marker: {color: '#007BFF'}
                        }];
                        var layout = {
                            title: 'Memory Usage by Service',
                            xaxis: {title: 'Service'},
                            yaxis: {title: 'Memory Used (MB)'}
                        };
                        Plotly.newPlot('service-memory-chart', data, layout);
                    </script>
                    """ % (json.dumps(service_memory['service_name'].tolist()), 
                          json.dumps(service_memory['max_memory_used'].tolist())))
                else:
                    f.write('<p>No service memory data available</p>')
            else:
                f.write('<p>Service information not available</p>')
            f.write('</div>\n')
            
            # 3. GPU利用率趋势
            f.write('<div class="chart-container">\n')
            f.write('<h2>GPU Utilization Over Time</h2>\n')
            if os.path.exists(gpu_path):
                gpu_df = pd.read_csv(gpu_path)
                if 'timestamp' in gpu_df:
                    gpu_df['timestamp'] = pd.to_datetime(gpu_df['timestamp'], unit='s')
                    gpu_df = gpu_df.sort_values('timestamp')
                    gpu_df = gpu_df.groupby('gpu_index')
                    
                    f.write('<div id="gpu-utilization-chart"></div>\n')
                    f.write('<script>\n')
                    f.write('var data = [];\n')
                    
                    for gpu_idx, group in gpu_df:
                        f.write(f"data.push({{name: 'GPU {gpu_idx}',\n")
                        f.write(f"x: {json.dumps(group['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist())},\n")
                        f.write(f"y: {json.dumps(group['avg_gpu_util'].tolist())},\n")
                        f.write("mode: 'lines+markers',\n")
                        f.write("type: 'scatter'});\n")
                    
                    f.write("""
                    var layout = {
                        title: 'GPU Utilization Over Time',
                        xaxis: {title: 'Time'},
                        yaxis: {title: 'Utilization (%)', range: [0, 100]},
                        legend: {orientation: 'h'}
                    };
                    Plotly.newPlot('gpu-utilization-chart', data, layout);
                    </script>
                    """)
                else:
                    f.write('<p>Timestamp information not available in GPU data</p>')
            else:
                f.write('<p>No GPU utilization data available</p>')
            f.write('</div>\n')
            
            # 4. 容器GPU时长
            f.write('<div class="chart-container">\n')
            f.write('<h2>GPU Usage Duration</h2>\n')
            if not container_df.empty:
                duration_df = container_df.groupby('container_name')['total_gpu_seconds'].sum().sort_values(ascending=False).head(10)
                if not duration_df.empty:
                    duration_df = duration_df.reset_index()
                    duration_df['hours'] = duration_df['total_gpu_seconds'] / 3600
                    
                    f.write('<div id="usage-duration-chart"></div>\n')
                    f.write("""
                    <script>
                        var data = [{
                            type: 'bar',
                            x: %s,
                            y: %s,
                            marker: {color: '#28A745'}
                        }];
                        var layout = {
                            title: 'Total GPU Usage (Hours)',
                            xaxis: {title: 'Container'},
                            yaxis: {title: 'Hours'}
                        };
                        Plotly.newPlot('usage-duration-chart', data, layout);
                    </script>
                    """ % (json.dumps(duration_df['container_name'].tolist()), 
                          json.dumps(duration_df['hours'].tolist())))
                else:
                    f.write('<p>No usage duration data available</p>')
            else:
                f.write('<p>No container data available</p>')
            f.write('</div>\n')
            

            # 添加甘特图
            f.write('<div class="chart-container" style="grid-column: span 2;">\n')
            f.write('<h2>Container GPU Usage Timeline</h2>\n')
            
            if container_timeline:
                # 构建甘特图数据
                f.write('<div id="timeline-chart"></div>\n')
                
                # 准备JavaScript数据
                f.write('<script>\n')
                f.write('var timelineData = [\n')
                
                # 为每个容器创建数据
                container_list = list(container_timeline.keys())
                for i, container in enumerate(container_list):
                    f.write('  {\n')
                    f.write(f'    container: "{container}",\n')
                    f.write('    tasks: [\n')
                    
                    # 添加时间段
                    for period in container_timeline[container]:
                        start_iso = period["start"].isoformat()
                        end_iso = period["end"].isoformat()
                        duration_hours = round(period["duration"] / 3600, 2)
                        
                        f.write('      {\n')
                        f.write(f'        start: "{start_iso}",\n')
                        f.write(f'        end: "{end_iso}",\n')
                        f.write(f'        duration: {duration_hours}\n')
                        f.write('      },\n')
                    
                    f.write('    ]\n')
                    f.write('  },\n')
                
                f.write('];\n\n')
                
                # 构建甘特图
                f.write("""
                var tasks = [];
                var yLabels = [];
                
                // 组织数据
                for (var i = 0; i < timelineData.length; i++) {
                    var container = timelineData[i];
                    yLabels.push(container.container);
                    
                    for (var j = 0; j < container.tasks.length; j++) {
                        var task = container.tasks[j];
                        tasks.push({
                            x: [task.start, task.end],
                            y: [container.container, container.container],
                            mode: 'lines',
                            line: {width: 20},
                            name: container.container,
                            showlegend: false,
                            hovertemplate: '<b>Container:</b> ' + container.container + 
                                '<br><b>Duration:</b> ' + task.duration.toFixed(2) + ' hours' +
                                '<br><b>From:</b> %{x|%Y-%m-%d %H:%M:%S}' +
                                '<br><b>To:</b> %{text}<extra></extra>',
                            text: new Date(task.end).toLocaleString(),
                        });
                    }
                }
                
                // 创建图表
                var layout = {
                    title: 'Container GPU Usage Timeline',
                    xaxis: {title: 'Time', type: 'date'},
                    yaxis: {
                        title: 'Container',
                        categoryorder: 'array',
                        categoryarray: yLabels.reverse(),
                        automargin: true
                    },
                    height: 600,
                    hovermode: 'closest',
                    margin: {l: 200}
                };
                
                Plotly.newPlot('timeline-chart', tasks, layout);
                </script>
                """)
            else:
                f.write('<p>No timeline data available</p>')
                
            f.write('</div>\n')

            # 关闭HTML标签
            f.write("""
                </div>
                <script>
                    // 自动刷新页面
                    setTimeout(function() {
                        location.reload();
                    }, 60000); // 每分钟刷新一次
                </script>
            </body>
            </html>
            """)
            
        print(f"仪表盘已生成: file://{os.path.abspath(dashboard_path)}")
        return dashboard_path
    except Exception as e:
        print(f"生成仪表盘时出错: {e}")
        return None

# ===== Flask API 路由 =====

@app.route('/')
def index():
    """返回仪表盘主页"""
    dashboard_path = os.path.join(DATA_DIR, "gpu_dashboard.html")
    if os.path.exists(dashboard_path):
        return send_from_directory(DATA_DIR, "gpu_dashboard.html")
    else:
        return "仪表盘尚未生成，请等待监控系统运行一段时间后再访问"

@app.route('/api/gpu/status')
def get_gpu_status():
    """获取当前GPU状态"""
    try:
        pynvml.nvmlInit()
        gpu_info = []
        
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # 获取GPU名称
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            
            # 获取内存信息
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_mem = mem_info.total // (1024 * 1024)  # MB
            used_mem = mem_info.used // (1024 * 1024)    # MB
            free_mem = mem_info.free // (1024 * 1024)    # MB
            mem_percent = (used_mem / total_mem) * 100 if total_mem > 0 else 0
            
            # 获取温度和利用率
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu if hasattr(util, 'gpu') else 0
            
            # 获取运行中的进程
            try:
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                process_info = []
                for proc in processes:
                    process_info.append({
                        "pid": proc.pid,
                        "used_memory_mb": proc.usedGpuMemory // (1024 * 1024) if hasattr(proc, 'usedGpuMemory') else 0
                    })
            except pynvml.NVMLError:
                process_info = []
            
            gpu_info.append({
                "index": i,
                "name": name,
                "temperature": temp,
                "memory": {
                    "total": total_mem,
                    "used": used_mem,
                    "free": free_mem,
                    "percent": mem_percent
                },
                "utilization": gpu_util,
                "processes": process_info
            })
        
        pynvml.nvmlShutdown()
        return jsonify({
            "status": "success",
            "timestamp": time.time(),
            "data": gpu_info
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/containers')
def get_containers():
    """获取当前GPU容器信息"""
    try:
        gpu_processes = get_gpu_processes()
        container_data = []
        
        for proc in gpu_processes:
            container_id, container_name, service_name = get_container_info(proc["pid"])
            
            if container_id == "host":
                continue
                
            container_data.append({
                "container_id": container_id,
                "container_name": container_name,
                "service_name": service_name,
                "gpu_index": proc["gpu_index"],
                "memory_usage_mb": proc["used_memory_mb"],
                "memory_percent": proc["memory_percent"],
                "gpu_utilization": proc["gpu_utilization"],
                "pid": proc["pid"],
                "command": proc["command"]
            })
        
        return jsonify({
            "status": "success",
            "timestamp": time.time(),
            "data": container_data
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/refresh')
def refresh_dashboard():
    """手动触发仪表盘刷新"""
    try:
        aggregate_data()
        dashboard_path = generate_dashboard()
        return jsonify({
            "status": "success",
            "message": "仪表盘已刷新",
            "dashboard_path": dashboard_path
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# ===== 监控线程 =====

class MonitorThread(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self._stop_event = threading.Event()
        
    def stop(self):
        self._stop_event.set()
        
    def run(self):
        print("启动GPU监控线程...")
        last_aggregation = time.time()
        
        while not self._stop_event.is_set():
            try:
                # 获取当前时间
                now = time.time()
                
                # 收集GPU数据
                processes = get_gpu_processes()
                
                if processes:
                    save_raw_data(processes)
                
                # 定期聚合数据和生成报告
                if now - last_aggregation > AGGREGATION_INTERVAL:
                    print(f"聚合数据中...")
                    aggregate_data()
                    print(f"生成仪表盘中...")
                    generate_dashboard()
                    last_aggregation = now
                
                # 等待下一个收集周期
                print(f"等待下次采集({LOG_INTERVAL}秒)...")
                time.sleep(LOG_INTERVAL)
                
            except Exception as e:
                print(f"监控线程运行出错: {e}")
                time.sleep(LOG_INTERVAL)

# ===== 主程序 =====

if __name__ == "__main__":
    print(f"启动GPU监控系统 - 数据目录: {DATA_DIR}")
    print(f"Flask服务器将在端口 {FLASK_PORT} 运行")
    
    # 初始化GPU监控
    try:
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        print(f"检测到 {gpu_count} 个GPU设备")
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"GPU初始化失败: {e}")
    
    # 启动监控线程
    monitor_thread = MonitorThread()
    monitor_thread.start()
    
    # 启动Flask服务器
    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False)
    except KeyboardInterrupt:
        print("接收到中断信号，正在停止...")
        monitor_thread.stop()
        monitor_thread.join()
        print("监控系统已停止")