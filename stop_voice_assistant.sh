#!/bin/bash

# Stop Script for Voxtral + Orpheus Voice Assistant

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

echo "🛑 Stopping Voxtral + Orpheus Voice Assistant..."
echo "=============================================="

# Function to stop process by PID file
stop_by_pid_file() {
    if [ -f "voice_assistant.pid" ]; then
        local pid=$(cat voice_assistant.pid)
        print_step "Found PID file with PID: $pid"
        
        if ps -p $pid > /dev/null 2>&1; then
            print_step "Stopping process $pid..."
            kill $pid
            
            # Wait for graceful shutdown
            local count=0
            while ps -p $pid > /dev/null 2>&1 && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
                echo -n "."
            done
            echo ""
            
            if ps -p $pid > /dev/null 2>&1; then
                print_warning "Process didn't stop gracefully, forcing kill..."
                kill -9 $pid
                sleep 2
            fi
            
            if ! ps -p $pid > /dev/null 2>&1; then
                print_status "Process $pid stopped successfully"
                rm -f voice_assistant.pid
                return 0
            else
                print_error "Failed to stop process $pid"
                return 1
            fi
        else
            print_warning "PID $pid not running, removing stale PID file"
            rm -f voice_assistant.pid
            return 0
        fi
    else
        return 1
    fi
}

# Function to stop process by name
stop_by_name() {
    local pids=$(pgrep -f "voice_assistant.py" 2>/dev/null)
    
    if [ ! -z "$pids" ]; then
        print_step "Found voice assistant processes: $pids"
        
        for pid in $pids; do
            print_step "Stopping process $pid..."
            kill $pid
            
            # Wait for graceful shutdown
            local count=0
            while ps -p $pid > /dev/null 2>&1 && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
                echo -n "."
            done
            echo ""
            
            if ps -p $pid > /dev/null 2>&1; then
                print_warning "Process $pid didn't stop gracefully, forcing kill..."
                kill -9 $pid
                sleep 2
            fi
            
            if ! ps -p $pid > /dev/null 2>&1; then
                print_status "Process $pid stopped successfully"
            else
                print_error "Failed to stop process $pid"
            fi
        done
        return 0
    else
        return 1
    fi
}

# Function to stop processes using port 8555
stop_by_port() {
    local pids=$(lsof -t -i:8555 2>/dev/null)
    
    if [ ! -z "$pids" ]; then
        print_step "Found processes using port 8555: $pids"
        
        for pid in $pids; do
            print_step "Stopping process $pid using port 8555..."
            kill $pid
            sleep 2
            
            if ps -p $pid > /dev/null 2>&1; then
                print_warning "Process $pid didn't stop gracefully, forcing kill..."
                kill -9 $pid
                sleep 2
            fi
            
            if ! ps -p $pid > /dev/null 2>&1; then
                print_status "Process $pid stopped successfully"
            else
                print_error "Failed to stop process $pid"
            fi
        done
        return 0
    else
        return 1
    fi
}

# Try different methods to stop the application
stopped=false

# Method 1: Stop by PID file
if stop_by_pid_file; then
    stopped=true
fi

# Method 2: Stop by process name
if [ "$stopped" = false ]; then
    if stop_by_name; then
        stopped=true
    fi
fi

# Method 3: Stop by port usage
if [ "$stopped" = false ]; then
    if stop_by_port; then
        stopped=true
    fi
fi

# Check if anything was stopped
if [ "$stopped" = true ]; then
    print_status "✅ Voice assistant stopped successfully"
    
    # Clean up
    print_step "Cleaning up..."
    rm -f voice_assistant.pid
    
    # Clear GPU memory
    if command -v nvidia-smi &> /dev/null; then
        print_step "Clearing GPU memory..."
        python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU memory cleared')
" 2>/dev/null || true
    fi
    
    print_status "🎉 Cleanup completed"
else
    print_warning "No voice assistant processes found running"
fi

# Verify port is free
if lsof -Pi :8555 -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_error "Port 8555 is still in use!"
    print_step "Processes still using port 8555:"
    lsof -Pi :8555 -sTCP:LISTEN
else
    print_status "Port 8555 is now free"
fi

echo ""
print_status "🛑 Stop script completed"
