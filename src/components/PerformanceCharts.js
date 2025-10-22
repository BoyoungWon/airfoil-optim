import React, { useState } from 'react';
import Plot from 'react-plotly.js';

const PerformanceCharts = ({ analysisResults, xfoilConfig }) => {
  const [activeTab, setActiveTab] = useState('polar');

  const styles = {
    header: {
      fontSize: '18px',
      fontWeight: '600',
      marginBottom: '16px',
      color: '#1976d2',
      borderBottom: '2px solid #e3f2fd',
      paddingBottom: '8px'
    },
    tabContainer: {
      display: 'flex',
      marginBottom: '16px',
      borderBottom: '1px solid #e0e0e0'
    },
    tab: {
      padding: '8px 16px',
      cursor: 'pointer',
      border: 'none',
      background: 'transparent',
      fontSize: '13px',
      fontWeight: '500',
      color: '#666',
      borderBottom: '2px solid transparent',
      transition: 'all 0.2s ease'
    },
    activeTab: {
      color: '#1976d2',
      borderBottom: '2px solid #1976d2'
    },
    chartContainer: {
      height: '350px',
      border: '1px solid #e0e0e0',
      borderRadius: '6px',
      marginBottom: '16px'
    },
    metricsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
      gap: '12px',
      marginTop: '16px'
    },
    metric: {
      background: '#f8f9fa',
      border: '1px solid #dee2e6',
      borderRadius: '6px',
      padding: '12px',
      textAlign: 'center'
    },
    metricValue: {
      fontSize: '16px',
      fontWeight: '600',
      color: '#1976d2',
      marginBottom: '4px'
    },
    metricLabel: {
      fontSize: '11px',
      color: '#666',
      fontWeight: '500'
    },
    infoBox: {
      background: '#e8f5e8',
      border: '1px solid #c8e6c9',
      borderRadius: '4px',
      padding: '8px',
      fontSize: '11px',
      color: '#2e7d32',
      marginTop: '12px'
    },
    convergenceInfo: {
      background: '#fff3e0',
      border: '1px solid #ffcc02',
      borderRadius: '4px',
      padding: '8px',
      fontSize: '11px',
      color: '#e65100',
      marginBottom: '12px'
    }
  };

  if (!analysisResults || !analysisResults.results) {
    return (
      <div>
        <h2 style={styles.header}>Performance Analysis Results</h2>
        <div style={{ ...styles.chartContainer, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <span style={{ color: '#999', fontSize: '14px' }}>
            No analysis results available. Run X-foil analysis first.
          </span>
        </div>
      </div>
    );
  }

  const results = analysisResults.results;
  const objectives = analysisResults.objectives || {};

  // 수렴한 데이터 포인트만 필터링
  const convergedIndices = results.converged?.map((conv, idx) => conv ? idx : null).filter(idx => idx !== null) || [];
  const convergedData = {
    alpha: convergedIndices.map(i => results.alpha[i]),
    cl: convergedIndices.map(i => results.cl[i]),
    cd: convergedIndices.map(i => results.cd[i]),
    cm: convergedIndices.map(i => results.cm[i])
  };

  // 극곡선 차트 데이터
  const getPolarData = () => [{
    x: convergedData.cd,
    y: convergedData.cl,
    type: 'scatter',
    mode: 'lines+markers',
    name: 'Cl vs Cd',
    marker: {
      color: convergedData.alpha,
      colorscale: 'Viridis',
      size: 8,
      colorbar: {
        title: 'α (deg)',
        titleside: 'right'
      }
    },
    line: { color: '#1976d2', width: 2 },
    hovertemplate: 'Cd: %{x:.6f}<br>Cl: %{y:.4f}<br>α: %{marker.color:.1f}°<extra></extra>'
  }];

  // 받음각 vs 계수 차트 데이터
  const getAlphaData = () => [
    {
      x: convergedData.alpha,
      y: convergedData.cl,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Cl',
      line: { color: '#4caf50', width: 2 },
      marker: { color: '#4caf50', size: 6 },
      hovertemplate: 'α: %{x:.1f}°<br>Cl: %{y:.4f}<extra></extra>'
    },
    {
      x: convergedData.alpha,
      y: convergedData.cd,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Cd',
      yaxis: 'y2',
      line: { color: '#f44336', width: 2 },
      marker: { color: '#f44336', size: 6 },
      hovertemplate: 'α: %{x:.1f}°<br>Cd: %{y:.6f}<extra></extra>'
    },
    {
      x: convergedData.alpha,
      y: convergedData.cm,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Cm',
      yaxis: 'y3',
      line: { color: '#ff9800', width: 2 },
      marker: { color: '#ff9800', size: 6 },
      hovertemplate: 'α: %{x:.1f}°<br>Cm: %{y:.4f}<extra></extra>'
    }
  ];

  // L/D 비 차트 데이터
  const getLiftDragData = () => {
    const liftDragRatio = convergedData.cl.map((cl, i) => 
      convergedData.cd[i] > 0.001 ? cl / convergedData.cd[i] : 0
    );

    return [{
      x: convergedData.alpha,
      y: liftDragRatio,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'L/D Ratio',
      line: { color: '#9c27b0', width: 2 },
      marker: { color: '#9c27b0', size: 6 },
      hovertemplate: 'α: %{x:.1f}°<br>L/D: %{y:.1f}<extra></extra>'
    }];
  };

  // 차트 레이아웃 설정
  const getLayout = (title, xTitle, yTitle, yTitle2 = null, yTitle3 = null) => {
    const layout = {
      title: { text: title, font: { size: 14 } },
      xaxis: { 
        title: xTitle,
        showgrid: true,
        gridcolor: 'rgba(0,0,0,0.1)'
      },
      yaxis: { 
        title: yTitle,
        showgrid: true,
        gridcolor: 'rgba(0,0,0,0.1)'
      },
      margin: { t: 40, b: 60, l: 60, r: 60 },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white',
      hovermode: 'closest',
      showlegend: true,
      legend: {
        x: 1.02,
        y: 1,
        bgcolor: 'rgba(255,255,255,0.9)',
        bordercolor: 'rgba(0,0,0,0.1)',
        borderwidth: 1,
        font: { size: 10 }
      }
    };

    if (yTitle2) {
      layout.yaxis2 = {
        title: yTitle2,
        overlaying: 'y',
        side: 'right',
        showgrid: false
      };
    }

    if (yTitle3) {
      layout.yaxis3 = {
        title: yTitle3,
        overlaying: 'y',
        side: 'right',
        position: 0.85,
        showgrid: false
      };
    }

    return layout;
  };

  const config = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d'],
    responsive: true
  };

  // 주요 성능 지표 계산
  const calculateMetrics = () => {
    if (convergedData.cl.length === 0) return {};

    const maxClIndex = convergedData.cl.indexOf(Math.max(...convergedData.cl));
    const maxCl = convergedData.cl[maxClIndex];
    const minCd = Math.min(...convergedData.cd);
    const maxLiftDragRatio = Math.max(...convergedData.cl.map((cl, i) => 
      convergedData.cd[i] > 0.001 ? cl / convergedData.cd[i] : 0
    ));

    // Cl = 1.0일 때의 Cd 찾기
    let cdAtCl1 = null;
    for (let i = 0; i < convergedData.cl.length - 1; i++) {
      const cl1 = convergedData.cl[i];
      const cl2 = convergedData.cl[i + 1];
      if ((cl1 <= 1.0 && cl2 >= 1.0) || (cl1 >= 1.0 && cl2 <= 1.0)) {
        const t = (1.0 - cl1) / (cl2 - cl1);
        cdAtCl1 = convergedData.cd[i] + t * (convergedData.cd[i + 1] - convergedData.cd[i]);
        break;
      }
    }

    return {
      maxCl,
      minCd,
      maxLiftDragRatio,
      cdAtCl1,
      stallAngle: convergedData.alpha[maxClIndex],
      convergedPoints: convergedData.alpha.length,
      totalPoints: results.alpha.length
    };
  };

  const metrics = calculateMetrics();

  const renderChart = () => {
    switch (activeTab) {
      case 'polar':
        return (
          <Plot
            data={getPolarData()}
            layout={getLayout('Airfoil Polar Curve', 'Cd', 'Cl')}
            config={config}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler={true}
          />
        );
      case 'alpha':
        return (
          <Plot
            data={getAlphaData()}
            layout={getLayout('Coefficients vs Angle of Attack', 'Angle of Attack (°)', 'Cl', 'Cd', 'Cm')}
            config={config}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler={true}
          />
        );
      case 'liftdrag':
        return (
          <Plot
            data={getLiftDragData()}
            layout={getLayout('Lift-to-Drag Ratio', 'Angle of Attack (°)', 'L/D Ratio')}
            config={config}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler={true}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div>
      <h2 style={styles.header}>Performance Analysis Results</h2>

      {/* 수렴 정보 */}
      {metrics.convergedPoints < metrics.totalPoints && (
        <div style={styles.convergenceInfo}>
          ⚠️ Convergence Warning: {metrics.convergedPoints}/{metrics.totalPoints} points converged. 
          Consider adjusting Reynolds number or reducing angle range.
        </div>
      )}

      {/* 탭 네비게이션 */}
      <div style={styles.tabContainer}>
        {[
          { key: 'polar', label: 'Polar Curve' },
          { key: 'alpha', label: 'α vs Coefficients' },
          { key: 'liftdrag', label: 'L/D Ratio' }
        ].map(tab => (
          <button
            key={tab.key}
            style={{
              ...styles.tab,
              ...(activeTab === tab.key ? styles.activeTab : {})
            }}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* 차트 */}
      <div style={styles.chartContainer}>
        {renderChart()}
      </div>

      {/* 성능 지표 */}
      <div style={styles.metricsGrid}>
        <div style={styles.metric}>
          <div style={styles.metricValue}>{metrics.maxCl?.toFixed(3) || 'N/A'}</div>
          <div style={styles.metricLabel}>Max Cl</div>
        </div>
        <div style={styles.metric}>
          <div style={styles.metricValue}>{metrics.minCd?.toFixed(5) || 'N/A'}</div>
          <div style={styles.metricLabel}>Min Cd</div>
        </div>
        <div style={styles.metric}>
          <div style={styles.metricValue}>{metrics.maxLiftDragRatio?.toFixed(1) || 'N/A'}</div>
          <div style={styles.metricLabel}>Max L/D</div>
        </div>
        <div style={styles.metric}>
          <div style={styles.metricValue}>{metrics.cdAtCl1?.toFixed(5) || 'N/A'}</div>
          <div style={styles.metricLabel}>Cd @ Cl=1.0</div>
        </div>
        <div style={styles.metric}>
          <div style={styles.metricValue}>{metrics.stallAngle?.toFixed(1) || 'N/A'}°</div>
          <div style={styles.metricLabel}>Stall Angle</div>
        </div>
        <div style={styles.metric}>
          <div style={styles.metricValue}>{objectives.min_dcl_dcm?.toFixed(2) || 'N/A'}</div>
          <div style={styles.metricLabel}>dCl/dCm</div>
        </div>
      </div>

      {/* X-foil 설정 정보 */}
      <div style={styles.infoBox}>
        <strong>Analysis Settings:</strong> Re = {xfoilConfig.reynolds.toLocaleString()}, 
        M = {xfoilConfig.mach.toFixed(2)}, Ncrit = {xfoilConfig.ncrit}, 
        Max Iter = {xfoilConfig.maxIter}, 
        Viscous = {xfoilConfig.viscous ? 'Yes' : 'No'}
      </div>
    </div>
  );
};

export default PerformanceCharts;
