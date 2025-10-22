import React, { useEffect, useRef } from 'react';
import Plot from 'react-plotly.js';

const AirfoilVisualization = ({ 
  coordinates, 
  upperControlPoints, 
  lowerControlPoints, 
  showControlPoints = true 
}) => {
  const styles = {
    header: {
      fontSize: '18px',
      fontWeight: '600',
      marginBottom: '16px',
      color: '#1976d2',
      borderBottom: '2px solid #e3f2fd',
      paddingBottom: '8px'
    },
    container: {
      position: 'relative',
      height: '400px',
      border: '1px solid #e0e0e0',
      borderRadius: '6px',
      overflow: 'hidden'
    },
    controls: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '12px'
    },
    toggleButton: {
      padding: '4px 8px',
      fontSize: '11px',
      border: '1px solid #e0e0e0',
      borderRadius: '4px',
      background: 'white',
      cursor: 'pointer',
      transition: 'all 0.2s ease'
    },
    toggleButtonActive: {
      background: '#1976d2',
      color: 'white',
      borderColor: '#1976d2'
    },
    info: {
      fontSize: '12px',
      color: '#666'
    },
    placeholder: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100%',
      color: '#999',
      fontSize: '14px',
      background: '#fafafa'
    }
  };

  const [showCP, setShowCP] = React.useState(showControlPoints);

  // Plotly 데이터 생성
  const generatePlotData = () => {
    const data = [];

    // 에어포일 곡선
    if (coordinates && coordinates.length > 0) {
      const x = coordinates.map(coord => coord[0]);
      const y = coordinates.map(coord => coord[1]);

      data.push({
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines',
        name: 'Airfoil',
        line: {
          color: '#1976d2',
          width: 3
        },
        fill: 'toself',
        fillcolor: 'rgba(25, 118, 210, 0.1)',
        hovertemplate: 'x/c: %{x:.4f}<br>y/c: %{y:.4f}<extra></extra>'
      });
    }

    // 상면 제어점
    if (showCP && upperControlPoints && upperControlPoints.length > 0) {
      const upperX = upperControlPoints.map(cp => cp[0]);
      const upperY = upperControlPoints.map(cp => cp[1]);

      data.push({
        x: upperX,
        y: upperY,
        type: 'scatter',
        mode: 'markers+lines',
        name: 'Upper Control Points',
        marker: {
          color: '#f44336',
          size: 8,
          symbol: 'circle',
          line: {
            color: 'white',
            width: 2
          }
        },
        line: {
          color: '#f44336',
          width: 1,
          dash: 'dot'
        },
        hovertemplate: 'CP: (%{x:.4f}, %{y:.4f})<extra></extra>'
      });
    }

    // 하면 제어점
    if (showCP && lowerControlPoints && lowerControlPoints.length > 0) {
      const lowerX = lowerControlPoints.map(cp => cp[0]);
      const lowerY = lowerControlPoints.map(cp => cp[1]);

      data.push({
        x: lowerX,
        y: lowerY,
        type: 'scatter',
        mode: 'markers+lines',
        name: 'Lower Control Points',
        marker: {
          color: '#4caf50',
          size: 8,
          symbol: 'circle',
          line: {
            color: 'white',
            width: 2
          }
        },
        line: {
          color: '#4caf50',
          width: 1,
          dash: 'dot'
        },
        hovertemplate: 'CP: (%{x:.4f}, %{y:.4f})<extra></extra>'
      });
    }

    return data;
  };

  // Plotly 레이아웃 설정
  const layout = {
    title: {
      text: 'Airfoil Geometry (Chord = 1.0)',
      font: { size: 16, color: '#424242' }
    },
    xaxis: {
      title: { text: 'x/c', font: { size: 12 } },
      range: [-0.05, 1.05],
      scaleanchor: 'y',
      scaleratio: 1,
      showgrid: true,
      gridcolor: 'rgba(0,0,0,0.1)',
      zeroline: true,
      zerolinecolor: 'rgba(0,0,0,0.3)'
    },
    yaxis: {
      title: { text: 'y/c', font: { size: 12 } },
      range: [-0.3, 0.3],
      showgrid: true,
      gridcolor: 'rgba(0,0,0,0.1)',
      zeroline: true,
      zerolinecolor: 'rgba(0,0,0,0.3)'
    },
    showlegend: true,
    legend: {
      x: 1,
      y: 1,
      bgcolor: 'rgba(255,255,255,0.9)',
      bordercolor: 'rgba(0,0,0,0.1)',
      borderwidth: 1,
      font: { size: 10 }
    },
    margin: { t: 60, b: 60, l: 80, r: 120 },
    plot_bgcolor: 'white',
    paper_bgcolor: 'white',
    hovermode: 'closest',
    dragmode: 'pan'
  };

  // Plotly 설정
  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d', 'autoScale2d'],
    modeBarButtonsToAdd: [
      {
        name: 'Export as PNG',
        icon: {
          width: 857.1,
          height: 1000,
          path: 'M214.3 714.3c0 40.8 33.1 73.9 73.9 73.9s73.9-33.1 73.9-73.9c0-40.8-33.1-73.9-73.9-73.9S214.3 673.5 214.3 714.3z M214.3 285.7c0 40.8 33.1 73.9 73.9 73.9s73.9-33.1 73.9-73.9c0-40.8-33.1-73.9-73.9-73.9S214.3 244.9 214.3 285.7z M428.6 714.3c0 40.8 33.1 73.9 73.9 73.9s73.9-33.1 73.9-73.9c0-40.8-33.1-73.9-73.9-73.9S428.6 673.5 428.6 714.3z M642.9 714.3c0 40.8 33.1 73.9 73.9 73.9s73.9-33.1 73.9-73.9c0-40.8-33.1-73.9-73.9-73.9S642.9 673.5 642.9 714.3z M428.6 285.7c0 40.8 33.1 73.9 73.9 73.9s73.9-33.1 73.9-73.9c0-40.8-33.1-73.9-73.9-73.9S428.6 244.9 428.6 285.7z M642.9 285.7c0 40.8 33.1 73.9 73.9 73.9s73.9-33.1 73.9-73.9c0-40.8-33.1-73.9-73.9-73.9S642.9 244.9 642.9 285.7z',
          transform: 'matrix(1 0 0 -1 0 850)'
        },
        click: function(gd) {
          Plotly.downloadImage(gd, {format: 'png', width: 800, height: 600, filename: 'airfoil'});
        }
      }
    ],
    responsive: true
  };

  const plotData = generatePlotData();

  return (
    <div>
      <div style={styles.controls}>
        <h2 style={styles.header}>Airfoil Geometry</h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={styles.info}>
            Points: {coordinates?.length || 0} | Chord: 1.0
          </div>
          <button
            style={{
              ...styles.toggleButton,
              ...(showCP ? styles.toggleButtonActive : {})
            }}
            onClick={() => setShowCP(!showCP)}
          >
            {showCP ? '👁️ Hide CP' : '👁️ Show CP'}
          </button>
        </div>
      </div>

      <div style={styles.container}>
        {plotData.length > 0 ? (
          <Plot
            data={plotData}
            layout={layout}
            config={config}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler={true}
          />
        ) : (
          <div style={styles.placeholder}>
            No airfoil data available. Click "Run X-foil Analysis" to load the initial airfoil.
          </div>
        )}
      </div>
    </div>
  );
};

export default AirfoilVisualization;