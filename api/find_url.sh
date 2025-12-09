#!/bin/bash
# Script to find the correct URL for accessing the API

echo "=== Finding Access URL ==="
echo ""

# 1. Check server IP
echo "1. Server IP addresses:"
hostname -I
echo ""

# 2. Check if port 8000 is listening
echo "2. Checking if port 8000 is listening:"
netstat -tuln | grep 8000 || ss -tuln | grep 8000
echo ""

# 3. Check Lightning AI environment variables
echo "3. Lightning AI environment variables:"
env | grep -i lightning || echo "No Lightning AI env vars found"
echo ""

# 4. Check if there's a public URL file
echo "4. Checking for public URL info:"
if [ -f ~/.lightning/url.txt ]; then
    cat ~/.lightning/url.txt
elif [ -n "$LIGHTNING_URL" ]; then
    echo "LIGHTNING_URL: $LIGHTNING_URL"
else
    echo "Check your Lightning AI dashboard for the public URL"
fi
echo ""

# 5. Test localhost connection
echo "5. Testing localhost connection:"
curl -s -o /dev/null -w "HTTP Status: %{http_code}\n" http://localhost:8000/ || echo "Connection failed"
echo ""

# 6. Get external IP (if accessible)
echo "6. External IP (if available):"
curl -s ifconfig.me || curl -s icanhazip.com || echo "Cannot determine external IP"
echo ""

echo "=== Instructions ==="
echo "1. Check your Lightning AI dashboard for the public URL"
echo "2. The URL should be something like: https://xxxxx.lightning.ai"
echo "3. Access the UI at: https://xxxxx.lightning.ai:8000/"
echo "4. Or use the server IP if you have direct VPN access: http://<SERVER_IP>:8000/"

