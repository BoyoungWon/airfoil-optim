import React, { useState } from 'react';
import Plot from 'react-plotly.js';

const OptimizationResults = ({ results, onSelectSolution }) => {
  const [selectedSolution, setSelectedSolution] = useState(null);
  const [showDetails, setShowDetails] = useState(false);

  const styles = {
    header: {
      fontSize: '18px',
      fontWeight: '600',
      marginBottom: '16px',
      color: '#1976d2',
      borderBottom: '2px solid #e3f2fd',
      paddingBottom: '8px'
    },
    summaryCard: {
      background: '#e8f5e8',
      border: '1px solid #c8e6c9',
      borderRadius: '6px',
      padding: '12px',
      marginBottom: '16px'
    },
    summaryGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(2, 1fr)',
      gap: '8px',
      fontSize: '12px'
    },
    summaryItem: {
      display: 'flex',
      justifyContent: 'space-between'
    },
    chartContainer: {
      height: '300px',
      border: '1px solid #e0e0e0',
      borderRadius: '6px',
      marginBottom: '16px'
    },
    solutionsList: {
      maxHeight: '200px',
      overflowY: 'auto',
      border: '1px solid #e0e0e0',
      borderRadius: '6px',
      marginBottom: '16px'
    },
    solutionItem: {
      padding: '8px 12px',
      borderBottom: '1px solid #f0f0f0',
      cursor: 'pointer',
      fontSize: '12px',
      transition: 'background-color 0.2s ease'
    },
    selectedSolutionItem: {
      backgroundColor: '#e3f2fd',
      borderColor: '#1976d2'
    },
    solutionHeader: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      fontWeight: '500',
      marginBottom: '4px'
    },
    solutionMetrics: {
      display: 'grid',
      gridTemplateColumns: 'repeat(3, 1fr)',
      gap: '8px',
      fontSize: '11px',
      color: '#666'
    },
    button: {
      padding: '8px 12px',
      border: 'none',
      borderRadius: '4px',
      fontSize: '12px',
      fontWeight: '500',
      cursor: 'pointer',
      transition: 'all 0.2s ease',
      marginRight: '8px'
    },
    primaryButton: {
      backgroundColor: '#1976d2',
      color: 'white'
    },
    secondaryButton: {
      backgroundColor: '#f5f5f5',
      color: '#666',
      border: '1px solid #e0e0e0'
    },
    detailsPanel: {
      background: '#f8f9fa',
      border: '1px solid #dee2e6',
      borderRadius: '6px',
      padding: '12px',
      marginTop: '12px'
    },
    detailsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
      gap: '12px'
    },
    detailCard: {
      background: 'white',
      padding: '8px',
      borderRadius: '4px',
      textAlign: 'center',
      border: '1px solid #e0e0e0'
    },
    detailValue: {
      fontSize: '14px',
      fontWeight: '600',
      color: '#1976d2',
      marginBottom: '2px'
    },
    detailLabel: {
      fontSize: '10px',
      color: '#666',
      textTransform: 'uppercase'
    },
    noResults: {
      textAlign: 'center',
      color: '#999',
      fontSize: '14px',
      padding: '40px'
    }
  };

  if (!results || !results.pareto_front || results.pareto_front.length === 0) {
    return (
      <div>
        <h2 style={styles.header}>Optimization Results</h2>
        <div style={styles.noResults}>
          No optimization results available yet.<br />
          Run optimization to see Pareto front solutions.
        </div>
      </div>
    );
  }

  const paretoFront = results.pareto_front;
  const convergenceHistory = results.convergence_history || [];

  // Pareto Front 3D 플롯 데이터
  const getParetoPlotData = () => {
    const clValues = paretoFront.map(sol => sol.objectives.cl);
    const cdValues = paretoFront.map(sol => sol.objectives.cd);
    const dclDcmValues = paretoFront.map(sol => sol.objectives.dcl_dcm);

    return [{
      x: cdValues,
      y: clValues,
      z: dclDcmValues,
      mode: 'markers',
      type: 'scatter3d',
      name: 'Pareto Solutions',
      marker: {
        size: 6,
        color: dclDcmValues,
        colorscale: 'Viridis',
        colorbar: {
          title: 'dCl/dCm',
          titleside: 'right'
        },
        line: {
          color: 'white',
          width: 1
        }
      },
      hovertemplate: 
        'Cd: %{x:.6f}<br>' +
        'Cl: %{y:.4f}<br>' +
        'dCl/dCm: %{z:.3f}<br>' +
        '<extra></extra>'
    }];
  };

  // 2D Pareto Front 플롯 데이터 (Cl vs Cd)
  const getPareto2DData = () => {
    const clValues = paretoFront.map(sol => sol.objectives.cl);
    const cdValues = paretoFront.map(sol => sol.objectives.cd);
    const dclDcmValues = paretoFront.map(sol => sol.objectives.dcl_dcm);

    return [{
      x: cdValues,
      y: clValues,
      mode: 'markers',
      type: 'scatter',
      name: 'Pareto Front',
      marker: {
        size: 8,
        color: dclDcmValues,
        colorscale: 'Viridis',
        colorbar: {
          title: 'dCl/dCm',
          titleside: 'right'
        },
        line: {
          color: 'white',
          width: 2
        }
      },
      hovertemplate: 
        'Cd: %{x:.6f}<br>' +
        'Cl: %{y:.4f}<br>' +
        'dCl/dCm: %{marker.color:.3f}<br>' +
        '<extra></extra>'
    }];
  };

  const plotLayout = {
    title: 'Pareto Front (Multi-Objective Solutions)',
    scene: {
      xaxis: { title: 'Cd (Drag Coefficient)' },
      yaxis: { title: 'Cl (Lift Coefficient)' },
      zaxis: { title: 'dCl/dCm' }
    },
    xaxis: { title: 'Cd (Drag Coefficient)' },
    yaxis: { title: 'Cl (Lift Coefficient)' },
    margin: { t: 40, b: 40, l: 60, r: 60 },
    hovermode: 'closest',
    showlegend: false
  };

  const plotConfig = {
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['select2d', 'lasso2d'],
    responsive: true
  };

  // 최적 솔루션들 분석
  const getBestSolutions = () => {
    const bestCl = paretoFront.reduce((best, current) => 
      current.objectives.cl > best.objectives.cl ? current : best);
    
    const bestCd = paretoFront.reduce((best, current) => 
      current.objectives.cd < best.objectives.cd ? current : best);
    
    const bestStability = paretoFront.reduce((best, current) => 
      current.objectives.dcl_dcm < best.objectives.dcl_dcm ? current : best);

    return { bestCl, bestCd, bestStability };
  };

  const bestSolutions = getBestSolutions();

  const handleSolutionSelect = (solution) => {
    setSelectedSolution(solution);
    onSelectSolution(solution);
  };

  const handleToggleDetails = () => {
    setShowDetails(!showDetails);
  };

  // 솔루션 품질 평가 (간단한 스코어링)
  const getSolutionScore = (solution) => {
    const normalizedCl = solution.objectives.cl / Math.max(...paretoFront.map(s => s.objectives.cl));
    const normalizedCd = 1 - (solution.objectives.cd / Math.max(...paretoFront.map(s => s.objectives.cd)));
    const normalizedStability = 1 - (solution.objectives.dcl_dcm / Math.max(...paretoFront.map(s => s.objectives.dcl_dcm)));
    
    return (normalizedCl + normalizedCd + normalizedStability) / 3;
  };

  // 솔루션을 점수순으로 정렬
  const sortedSolutions = [...paretoFront]
    .map(sol => ({ ...sol, score: getSolutionScore(sol) }))
    .sort((a, b) => b.score - a.score);

  return (
    <div>
      <h2 style={styles.header}>Optimization Results</h2>

      {/* 요약 정보 */}
      <div style={styles.summaryCard}>
        <div style={styles.summaryGrid}>
          <div style={styles.summaryItem}>
            <span>Pareto Solutions:</span>
            <strong>{paretoFront.length}</strong>
          </div>
          <div style={styles.summaryItem}>
            <span>Total Generations:</span>
            <strong>{results.total_generations || 0}</strong>
          </div>
          <div style={styles.summaryItem}>
            <span>Best Cl:</span>
            <strong>{bestSolutions.bestCl.objectives.cl.toFixed(4)}</strong>
          </div>
          <div style={styles.summaryItem}>
            <span>Best Cd:</span>
            <strong>{bestSolutions.bestCd.objectives.cd.toFixed(6)}</strong>
          </div>
        </div>
      </div>

      {/* Pareto Front 플롯 */}
      <div style={styles.chartContainer}>
        <Plot
          data={getPareto2DData()}
          layout={plotLayout}
          config={plotConfig}
          style={{ width: '100%', height: '100%' }}
          useResizeHandler={true}
        />
      </div>

      {/* 최고 성능 솔루션들 */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '8px' }}>
          Best Performing Solutions:
        </div>
        <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
          <button
            style={{ ...styles.button, ...styles.primaryButton }}
            onClick={() => handleSolutionSelect(bestSolutions.bestCl)}
          >
            Best Lift ({bestSolutions.bestCl.objectives.cl.toFixed(3)})
          </button>
          <button
            style={{ ...styles.button, ...styles.primaryButton }}
            onClick={() => handleSolutionSelect(bestSolutions.bestCd)}
          >
            Best Drag ({bestSolutions.bestCd.objectives.cd.toFixed(5)})
          </button>
          <button
            style={{ ...styles.button, ...styles.primaryButton }}
            onClick={() => handleSolutionSelect(bestSolutions.bestStability)}
          >
            Best Stability ({bestSolutions.bestStability.objectives.dcl_dcm.toFixed(3)})
          </button>
        </div>
      </div>

      {/* 솔루션 목록 */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '14px', fontWeight: '600' }}>
            All Solutions (Ranked by Overall Performance):
          </span>
          <button
            style={{ ...styles.button, ...styles.secondaryButton }}
            onClick={handleToggleDetails}
          >
            {showDetails ? 'Hide Details' : 'Show Details'}
          </button>
        </div>

        <div style={styles.solutionsList}>
          {sortedSolutions.map((solution, index) => (
            <div
              key={solution.id}
              style={{
                ...styles.solutionItem,
                ...(selectedSolution?.id === solution.id ? styles.selectedSolutionItem : {})
              }}
              onClick={() => handleSolutionSelect(solution)}
              onMouseEnter={(e) => {
                if (selectedSolution?.id !== solution.id) {
                  e.target.style.backgroundColor = '#f5f5f5';
                }
              }}
              onMouseLeave={(e) => {
                if (selectedSolution?.id !== solution.id) {
                  e.target.style.backgroundColor = 'transparent';
                }
              }}
            >
              <div style={styles.solutionHeader}>
                <span>Solution #{index + 1}</span>
                <span style={{ fontSize: '10px', color: '#1976d2' }}>
                  Score: {(solution.score * 100).toFixed(1)}%
                </span>
              </div>
              <div style={styles.solutionMetrics}>
                <span>Cl: {solution.objectives.cl.toFixed(4)}</span>
                <span>Cd: {solution.objectives.cd.toFixed(6)}</span>
                <span>dCl/dCm: {solution.objectives.dcl_dcm.toFixed(3)}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* 선택된 솔루션 상세 정보 */}
      {selectedSolution && showDetails && (
        <div style={styles.detailsPanel}>
          <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px' }}>
            Selected Solution Details:
          </div>
          
          <div style={styles.detailsGrid}>
            <div style={styles.detailCard}>
              <div style={styles.detailValue}>{selectedSolution.objectives.cl.toFixed(4)}</div>
              <div style={styles.detailLabel}>Lift Coefficient</div>
            </div>
            <div style={styles.detailCard}>
              <div style={styles.detailValue}>{selectedSolution.objectives.cd.toFixed(6)}</div>
              <div style={styles.detailLabel}>Drag Coefficient</div>
            </div>
            <div style={styles.detailCard}>
              <div style={styles.detailValue}>{selectedSolution.objectives.dcl_dcm.toFixed(3)}</div>
              <div style={styles.detailLabel}>dCl/dCm</div>
            </div>
            <div style={styles.detailCard}>
              <div style={styles.detailValue}>
                {(selectedSolution.objectives.cl / selectedSolution.objectives.cd).toFixed(1)}
              </div>
              <div style={styles.detailLabel}>L/D Ratio</div>
            </div>
            <div style={styles.detailCard}>
              <div style={styles.detailValue}>{selectedSolution.parameters.length}</div>
              <div style={styles.detailLabel}>Control Points</div>
            </div>
            <div style={styles.detailCard}>
              <div style={styles.detailValue}>
                {Math.max(...selectedSolution.parameters.map(Math.abs)).toFixed(4)}
              </div>
              <div style={styles.detailLabel}>Max Deformation</div>
            </div>
          </div>

          <div style={{ marginTop: '12px', textAlign: 'center' }}>
            <button
              style={{ ...styles.button, ...styles.primaryButton }}
              onClick={() => handleSolutionSelect(selectedSolution)}
            >
              Apply This Solution to Airfoil
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default OptimizationResults;
