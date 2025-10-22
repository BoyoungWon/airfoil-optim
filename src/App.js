import React, { useState, useEffect, useCallback } from 'react';

import XfoilConfigPanel from './components/XfoilConfigPanel';
import NurbsControlPanel from './components/NurbsControlPanel';
import AirfoilVisualization from './components/AirfoilVisualization';
import PerformanceCharts from './components/PerformanceCharts';
import OptimizationPanel from './components/OptimizationPanel';
import OptimizationResults from './components/OptimizationResults';
import useWebSocket from './hooks/useWebSocket';

// 앱 스타일
const appStyles = {
  header: {
    background: 'linear-gradient(135deg, #1976d2 0%, #42a5f5 100%)',
    color: 'white',
    padding: '16px 0',
    boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
  },
  headerContent: {
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '0 24px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between'
  },
  title: {
    fontSize: '28px',
    fontWeight: '600',
    margin: 0
  },
  subtitle: {
    fontSize: '14px',
    opacity: 0.9
  },
  container: {
    maxWidth: '1400px',
    margin: '0 auto',
    padding: '24px'
  },
  grid: {
    display: 'grid',
    gap: '24px',
    gridTemplateColumns: 'repeat(auto-fit, minmax(400px, 1fr))'
  },
  paper: {
    background: 'white',
    borderRadius: '12px',
    padding: '24px',
    boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
    border: '1px solid #e0e0e0'
  },
  fullWidth: {
    gridColumn: '1 / -1'
  },
  connectionChip: {
    padding: '4px 12px',
    borderRadius: '16px',
    fontSize: '12px',
    fontWeight: '500'
  },
  alert: {
    padding: '12px 16px',
    borderRadius: '8px',
    marginBottom: '16px',
    border: '1px solid #ff9800',
    background: '#fff3e0',
    color: '#e65100'
  }
};

const App = () => {
  // WebSocket 연결 상태
  const [socketUrl] = useState(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${window.location.host}/ws`;
  });

  const { connectionStatus, sendMessage, readyState } = useWebSocket(socketUrl);
  const [lastMessage, setLastMessage] = useState(null);

  // 애플리케이션 상태
  const [airfoilData, setAirfoilData] = useState({
    coordinates: [],
    upperControlPoints: [],
    lowerControlPoints: [],
    parameterBounds: { lower: [], upper: [] }
  });

  const [analysisResults, setAnalysisResults] = useState(null);
  const [optimizationResults, setOptimizationResults] = useState(null);
  const [optimizationProgress, setOptimizationProgress] = useState({
    isRunning: false,
    progress: 0,
    generation: 0,
    message: ''
  });

  const [controlParameters, setControlParameters] = useState([]);
  const [xfoilConfig, setXfoilConfig] = useState({
    reynolds: 1000000,
    mach: 0.0,
    ncrit: 9.0,
    alphaMin: 0.0,
    alphaMax: 4.0,
    alphaSteps: 9,
    maxIter: 200,
    viscous: true
  });

  // WebSocket 핸들러
  useEffect(() => {
    const handleMessage = (message) => {
      setLastMessage(message);
      handleWebSocketMessage(message);
    };

    // WebSocket 메시지 리스너 등록 (useWebSocket 훅에서 처리)
  }, []);

  // WebSocket 메시지 핸들러
  const handleWebSocketMessage = useCallback((message) => {
    switch (message.type) {
      case 'initialization':
        setAirfoilData({
          coordinates: message.data.coordinates,
          upperControlPoints: message.data.upper_control_points,
          lowerControlPoints: message.data.lower_control_points,
          parameterBounds: message.data.parameter_bounds
        });
        setControlParameters(new Array(message.data.parameter_bounds.lower.length).fill(0));
        break;

      case 'analysis_result':
        setAnalysisResults(message.data);
        break;

      case 'optimization_progress':
        setOptimizationProgress({
          isRunning: true,
          progress: message.progress || 0,
          generation: message.generation || 0,
          message: `Generation ${message.generation}: ${message.progress?.toFixed(1)}%`,
          bestObjectives: message.best_objectives
        });
        break;

      case 'optimization_complete':
        setOptimizationResults(message.data);
        setOptimizationProgress({
          isRunning: false,
          progress: 100,
          generation: message.data.total_generations || 0,
          message: 'Optimization completed'
        });
        break;

      case 'shape_update':
        setAirfoilData(prev => ({
          ...prev,
          coordinates: message.data.coordinates,
          upperControlPoints: message.data.upper_control_points,
          lowerControlPoints: message.data.lower_control_points
        }));
        break;

      case 'progress':
        setOptimizationProgress(prev => ({
          ...prev,
          progress: message.progress,
          message: message.message
        }));
        break;

      case 'error':
        console.error('WebSocket Error:', message.message);
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  }, []);

  // 초기화
  useEffect(() => {
    if (readyState === 'OPEN') {
      sendMessage(JSON.stringify({ action: 'initialize' }));
    }
  }, [readyState, sendMessage]);

  // 에어포일 해석 실행
  const handleAnalyzeAirfoil = useCallback(() => {
    if (readyState === WebSocket.OPEN) {
      const config = {
        alpha_min: xfoilConfig.alphaMin,
        alpha_max: xfoilConfig.alphaMax,
        alpha_steps: xfoilConfig.alphaSteps,
        reynolds: xfoilConfig.reynolds,
        mach: xfoilConfig.mach,
        ncrit: xfoilConfig.ncrit,
        max_iter: xfoilConfig.maxIter,
        viscous: xfoilConfig.viscous
      };

      sendMessage(JSON.stringify({
        action: 'analyze',
        config: config,
        parameters: controlParameters
      }));
    }
  }, [readyState, sendMessage, xfoilConfig, controlParameters]);

  // 최적화 실행
  const handleOptimizeAirfoil = useCallback((optimizationConfig) => {
    if (readyState === WebSocket.OPEN) {
      const config = {
        ...xfoilConfig,
        alpha_min: xfoilConfig.alphaMin,
        alpha_max: xfoilConfig.alphaMax,
        alpha_steps: xfoilConfig.alphaSteps,
        ...optimizationConfig
      };

      sendMessage(JSON.stringify({
        action: 'optimize',
        config: config,
        parameters: controlParameters
      }));

      setOptimizationProgress({
        isRunning: true,
        progress: 0,
        generation: 0,
        message: 'Starting optimization...'
      });
    }
  }, [readyState, sendMessage, xfoilConfig, controlParameters]);

  // 제어점 업데이트
  const handleUpdateControlPoints = useCallback((newParameters) => {
    setControlParameters(newParameters);
    
    if (readyState === WebSocket.OPEN) {
      sendMessage(JSON.stringify({
        action: 'update_control_points',
        parameters: newParameters
      }));
    }
  }, [readyState, sendMessage]);

  // 연결 상태 표시
  const getConnectionStatus = () => {
    const statusStyles = {
      Connected: { color: '#4caf50', label: '● Connected' },
      Connecting: { color: '#ff9800', label: '● Connecting...' },
      Disconnected: { color: '#f44336', label: '● Disconnected' },
      Error: { color: '#f44336', label: '● Connection Error' }
    };
    
    const status = statusStyles[connectionStatus] || statusStyles['Error'];
    return (
      <span style={{ ...appStyles.connectionChip, color: status.color }}>
        {status.label}
      </span>
    );
  };

  return (
    <div style={{ minHeight: '100vh', background: '#f5f5f5' }}>
      {/* 헤더 */}
      <div style={appStyles.header}>
        <div style={appStyles.headerContent}>
          <div>
            <h1 style={appStyles.title}>🛩️ X-foil Airfoil Optimizer</h1>
            <p style={appStyles.subtitle}>NURBS-based Airfoil Shape Optimization with CFD Analysis</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <span style={{ fontSize: '14px' }}>Re: {xfoilConfig.reynolds.toLocaleString()}</span>
            {getConnectionStatus()}
          </div>
        </div>
      </div>

      <div style={appStyles.container}>
        {/* 연결 상태 알림 */}
        {connectionStatus !== 'Connected' && (
          <div style={appStyles.alert}>
            WebSocket connection status: {connectionStatus}. 
            {connectionStatus === 'Disconnected' && ' Attempting to reconnect...'}
          </div>
        )}

        <div style={appStyles.grid}>
          {/* X-foil 설정 패널 */}
          <div style={appStyles.paper}>
            <XfoilConfigPanel
              config={xfoilConfig}
              onConfigChange={setXfoilConfig}
              onAnalyze={handleAnalyzeAirfoil}
              isAnalyzing={optimizationProgress.isRunning}
            />
          </div>

          {/* NURBS 제어 패널 */}
          <div style={appStyles.paper}>
            <NurbsControlPanel
              controlParameters={controlParameters}
              parameterBounds={airfoilData.parameterBounds}
              onParametersChange={handleUpdateControlPoints}
              disabled={optimizationProgress.isRunning}
            />
          </div>

          {/* 에어포일 시각화 */}
          <div style={{ ...appStyles.paper, ...appStyles.fullWidth }}>
            <AirfoilVisualization
              coordinates={airfoilData.coordinates}
              upperControlPoints={airfoilData.upperControlPoints}
              lowerControlPoints={airfoilData.lowerControlPoints}
              showControlPoints={true}
            />
          </div>

          {/* 성능 차트 */}
          {analysisResults && (
            <div style={{ ...appStyles.paper, ...appStyles.fullWidth }}>
              <PerformanceCharts
                analysisResults={analysisResults}
                xfoilConfig={xfoilConfig}
              />
            </div>
          )}

          {/* 최적화 패널 */}
          <div style={appStyles.paper}>
            <OptimizationPanel
              onOptimize={handleOptimizeAirfoil}
              progress={optimizationProgress}
              disabled={connectionStatus !== 'Connected'}
            />
          </div>

          {/* 최적화 결과 */}
          {optimizationResults && (
            <div style={appStyles.paper}>
              <OptimizationResults
                results={optimizationResults}
                onSelectSolution={(solution) => {
                  handleUpdateControlPoints(solution.parameters);
                }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;
};

export default App;