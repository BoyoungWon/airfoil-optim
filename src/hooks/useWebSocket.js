import { useState, useEffect, useCallback, useRef } from 'react';

const useWebSocket = (url) => {
  const [connectionStatus, setConnectionStatus] = useState('Connecting');
  const [lastMessage, setLastMessage] = useState(null);
  const ws = useRef(null);
  const reconnectTimeoutId = useRef(null);
  const messageQueue = useRef([]);

  // 연결 상태 관리
  const connect = useCallback(() => {
    try {
      setConnectionStatus('Connecting');
      
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        setConnectionStatus('Connected');
        console.log('WebSocket connected');
        
        // 대기 중인 메시지들 전송
        while (messageQueue.current.length > 0) {
          const message = messageQueue.current.shift();
          ws.current.send(message);
        }
      };

      ws.current.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          setLastMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };

      ws.current.onclose = (event) => {
        setConnectionStatus('Disconnected');
        console.log('WebSocket disconnected:', event.code, event.reason);
        
        // 자동 재연결 (1초 후)
        if (!event.wasClean) {
          reconnectTimeoutId.current = setTimeout(() => {
            console.log('Attempting to reconnect...');
            connect();
          }, 1000);
        }
      };

      ws.current.onerror = (error) => {
        setConnectionStatus('Error');
        console.error('WebSocket error:', error);
      };

    } catch (error) {
      setConnectionStatus('Error');
      console.error('Error creating WebSocket connection:', error);
    }
  }, [url]);

  // 메시지 전송
  const sendMessage = useCallback((message) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(message);
    } else {
      // 연결이 안 되어 있으면 큐에 저장
      messageQueue.current.push(message);
      
      if (ws.current && ws.current.readyState === WebSocket.CONNECTING) {
        // 연결 중이면 대기
        console.log('WebSocket connecting, message queued');
      } else {
        // 연결이 끊어져 있으면 재연결 시도
        console.log('WebSocket not connected, attempting to reconnect');
        connect();
      }
    }
  }, [connect]);

  // 연결 종료
  const disconnect = useCallback(() => {
    if (reconnectTimeoutId.current) {
      clearTimeout(reconnectTimeoutId.current);
    }
    
    if (ws.current) {
      ws.current.close(1000, 'Manual disconnect');
      ws.current = null;
    }
    
    setConnectionStatus('Disconnected');
  }, []);

  // 초기 연결 및 정리
  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // readyState 문자열 변환
  const getReadyState = () => {
    if (!ws.current) return 'CLOSED';
    
    const states = {
      [WebSocket.CONNECTING]: 'CONNECTING',
      [WebSocket.OPEN]: 'OPEN',
      [WebSocket.CLOSING]: 'CLOSING',
      [WebSocket.CLOSED]: 'CLOSED'
    };
    
    return states[ws.current.readyState] || 'UNKNOWN';
  };

  return {
    connectionStatus,
    lastMessage,
    sendMessage,
    disconnect,
    reconnect: connect,
    readyState: getReadyState()
  };
};

export default useWebSocket;
