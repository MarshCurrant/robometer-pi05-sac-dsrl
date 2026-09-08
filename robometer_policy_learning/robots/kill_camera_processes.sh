#!/bin/bash
# Script to kill processes that might be holding USB cameras

echo "Checking for processes using cameras..."

# Kill any droid remote server processes
pkill -f "droid_remote_server" && echo "✓ Killed droid_remote_server processes" || echo "  No droid_remote_server processes found"

# Kill any python processes that might be using cameras
pkill -f "python.*droid" && echo "✓ Killed python droid processes" || echo "  No python droid processes found"

# Kill any ZED SDK processes
pkill -f "ZED|stereolabs" && echo "✓ Killed ZED processes" || echo "  No ZED processes found"

# List remaining processes
echo ""
echo "Remaining processes that might use cameras:"
pgrep -f "droid|ZED|stereolabs" || echo "  No camera-related processes found"

echo ""
echo "If cameras are still locked, try:"
echo "1. Wait 2-3 seconds for USB devices to release"
echo "2. Unplug and replug the USB cameras"
echo "3. Or reset USB device (requires sudo):"
echo "   sudo usb_modeswitch -v 0x2b03 -p 0xf682 -R  # For ZED-M"
echo "   sudo usb_modeswitch -v 0x2b03 -p 0xf780 -R  # For ZED-2"
