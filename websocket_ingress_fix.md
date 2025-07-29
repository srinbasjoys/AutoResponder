# WebSocket Ingress Configuration Fix

## Problem
WebSocket connections are timing out during handshake phase due to missing Kubernetes ingress configuration for WebSocket support.

## Required Kubernetes Ingress Annotations

Add the following annotations to your Kubernetes ingress configuration:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: autoresponder-ingress
  annotations:
    # Enable WebSocket support
    nginx.ingress.kubernetes.io/websocket-services: "backend-service"
    
    # Increase timeouts for WebSocket connections
    nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "3600"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "3600"
    
    # Enable WebSocket connection upgrades
    nginx.ingress.kubernetes.io/proxy-set-header: "Upgrade $http_upgrade"
    nginx.ingress.kubernetes.io/proxy-set-header: "Connection $connection_upgrade"
    
    # Session affinity for WebSocket connections
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/session-cookie-name: "websocket-route"
    nginx.ingress.kubernetes.io/session-cookie-max-age: "3600"

spec:
  rules:
  - host: {your-domain}
    http:
      paths:
      - path: /ws
        pathType: Prefix
        backend:
          service:
            name: backend-service
            port:
              number: 8001
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-service
            port:
              number: 8001
```

## Alternative nginx.conf Configuration

If using a custom nginx configuration, add these settings:

```nginx
# Enable WebSocket support
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    location /ws {
        proxy_pass http://backend-service:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket specific timeouts
        proxy_connect_timeout 3600s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;
        
        # Disable buffering for WebSocket
        proxy_buffering off;
    }
}
```

## Testing WebSocket After Fix

Run the WebSocket test to verify the fix:

```bash
cd /app && python websocket_connectivity_test.py
```

Expected output after fix:
```
✅ HTTP connectivity successful
✅ WebSocket connected successfully
🔗 Connection established message received
🏓 Pong message received - WebSocket is fully functional!
```

## Impact of Fix

Once implemented, this will enable:
1. Real-time audio streaming
2. Live transcription updates
3. Streaming AI responses
4. Conversation interruption capabilities
5. True real-time conversation experience