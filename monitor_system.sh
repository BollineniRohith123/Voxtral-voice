#!/bin/bash

# System Monitor for Voxtral + Orpheus Voice Assistant
# Real-time monitoring of GPU, memory, and application status

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${CYAN}$1${NC}"
}

print_good() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

print_info() {
    echo -e "${BLUE}$1${NC}"
}

# Function to get GPU info
get_gpu_info() {
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null
    else
        echo "N/A,N/A,N/A,N/A,N/A,N/A"
    fi
}

# Function to get memory info
get_memory_info() {
    free -h | awk 'NR==2{printf "Used: %s/%s (%.1f%%)", $3, $2, $3/$2*100}'
}

# Function to get CPU info
get_cpu_info() {
    top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//'
}

# Function to check service health
check_health() {
    local response=$(curl -s -w "%{http_code}" http://localhost:8555/health -o /tmp/health_response 2>/dev/null)
    if [ "$response" = "200" ]; then
        local status=$(cat /tmp/health_response 2>/dev/null | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('status', 'unknown'))" 2>/dev/null)
        echo "✅ $status"
    else
        echo "❌ Service not responding"
    fi
    rm -f /tmp/health_response
}

# Function to get active connections
get_connections() {
    netstat -an 2>/dev/null | grep :8555 | grep ESTABLISHED | wc -l
}

# Function to get process info
get_process_info() {
    if [ -f "voice_assistant.pid" ]; then
        local pid=$(cat voice_assistant.pid)
        if ps -p $pid > /dev/null 2>&1; then
            local cpu_usage=$(ps -p $pid -o %cpu --no-headers 2>/dev/null | tr -d ' ')
            local mem_usage=$(ps -p $pid -o %mem --no-headers 2>/dev/null | tr -d ' ')
            echo "PID: $pid | CPU: ${cpu_usage}% | Memory: ${mem_usage}%"
        else
            echo "Process not running (stale PID file)"
        fi
    else
        local pid=$(pgrep -f "voice_assistant.py" 2>/dev/null | head -1)
        if [ ! -z "$pid" ]; then
            local cpu_usage=$(ps -p $pid -o %cpu --no-headers 2>/dev/null | tr -d ' ')
            local mem_usage=$(ps -p $pid -o %mem --no-headers 2>/dev/null | tr -d ' ')
            echo "PID: $pid | CPU: ${cpu_usage}% | Memory: ${mem_usage}%"
        else
            echo "Process not found"
        fi
    fi
}

# Function to get recent logs
get_recent_logs() {
    if [ -f "voice_assistant.log" ]; then
        tail -n 3 voice_assistant.log 2>/dev/null | while read line; do
            if [[ $line == *"ERROR"* ]]; then
                print_error "$line"
            elif [[ $line == *"WARNING"* ]]; then
                print_warning "$line"
            elif [[ $line == *"INFO"* ]]; then
                print_info "$line"
            else
                echo "$line"
            fi
        done
    elif [ -f "logs/voice_assistant.log" ]; then
        tail -n 3 logs/voice_assistant.log 2>/dev/null | while read line; do
            if [[ $line == *"ERROR"* ]]; then
                print_error "$line"
            elif [[ $line == *"WARNING"* ]]; then
                print_warning "$line"
            elif [[ $line == *"INFO"* ]]; then
                print_info "$line"
            else
                echo "$line"
            fi
        done
    else
        echo "No log file found"
    fi
}

# Function to display stats
display_stats() {
    local stats=$(curl -s http://localhost:8555/api/stats 2>/dev/null)
    if [ $? -eq 0 ] && [ ! -z "$stats" ]; then
        echo "$stats" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    server = data.get('server', {})
    models = data.get('models', {})
    
    print(f\"Active Connections: {server.get('active_connections', 'N/A')}\")
    print(f\"Total Requests: {server.get('total_requests', 'N/A')}\")
    print(f\"Audio Generated: {server.get('total_audio_generated', 'N/A')}\")
    print(f\"Uptime: {server.get('uptime', 0):.1f}s\")
    print(f\"Voxtral Loaded: {'✅' if models.get('voxtral_loaded') else '❌'}\")
    print(f\"Orpheus Loaded: {'✅' if models.get('orpheus_loaded') else '❌'}\")
except:
    print('Stats unavailable')
" 2>/dev/null
    else
        echo "Stats unavailable"
    fi
}

# Main monitoring loop
echo "📊 System Monitor for Voice Assistant"
echo "====================================="
echo "Press Ctrl+C to exit"
echo ""

# Trap Ctrl+C
trap 'echo -e "\n👋 Monitoring stopped."; exit 0' INT

while true; do
    clear
    
    # Header
    print_header "📊 Voice Assistant System Monitor - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=================================================================="
    
    # GPU Status
    echo ""
    print_header "🖥️  GPU Status:"
    gpu_info=$(get_gpu_info)
    IFS=',' read -r gpu_name gpu_mem_used gpu_mem_total gpu_util gpu_temp gpu_power <<< "$gpu_info"
    
    if [ "$gpu_name" != "N/A" ]; then
        echo "   Name: $gpu_name"
        echo "   Memory: ${gpu_mem_used}MB / ${gpu_mem_total}MB"
        echo "   Utilization: ${gpu_util}%"
        echo "   Temperature: ${gpu_temp}°C"
        echo "   Power: ${gpu_power}W"
        
        # Memory usage warning
        if [ "$gpu_mem_used" != "N/A" ] && [ "$gpu_mem_total" != "N/A" ]; then
            local usage_percent=$((gpu_mem_used * 100 / gpu_mem_total))
            if [ $usage_percent -gt 90 ]; then
                print_warning "   ⚠️  High GPU memory usage: ${usage_percent}%"
            fi
        fi
    else
        print_error "   GPU not available"
    fi
    
    # System Memory
    echo ""
    print_header "💾 System Memory:"
    echo "   $(get_memory_info)"
    
    # CPU Usage
    echo ""
    print_header "🔄 CPU Usage:"
    echo "   $(get_cpu_info)%"
    
    # Process Information
    echo ""
    print_header "🔧 Process Information:"
    echo "   $(get_process_info)"
    
    # Service Health
    echo ""
    print_header "🏥 Service Health:"
    echo "   $(check_health)"
    
    # Network Connections
    echo ""
    print_header "🌐 Network:"
    echo "   Active connections: $(get_connections)"
    
    # Application Stats
    echo ""
    print_header "📈 Application Stats:"
    display_stats | sed 's/^/   /'
    
    # Recent Logs
    echo ""
    print_header "📝 Recent Logs:"
    get_recent_logs | sed 's/^/   /'
    
    # URLs
    echo ""
    print_header "🔗 Access URLs:"
    echo "   Web Interface: http://localhost:8555"
    echo "   Health Check: http://localhost:8555/health"
    echo "   API Docs: http://localhost:8555/docs"
    
    if [ ! -z "$RUNPOD_PUBLIC_IP" ]; then
        echo "   Public URL: https://$RUNPOD_PUBLIC_IP-8555.proxy.runpod.net/"
    fi
    
    echo ""
    print_info "🔄 Refreshing in 5 seconds... (Press Ctrl+C to exit)"
    
    sleep 5
done
