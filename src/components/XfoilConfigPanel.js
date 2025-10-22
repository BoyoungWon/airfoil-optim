import React, { useState } from 'react';

const OptimizationPanel = ({ onOptimize, progress, disabled }) => {
  const [config, setConfig] = useState({
    algorithm: 'nsga2',
    populationSize: 20,
    maxGenerations: 50,
    objectives: {
      maxCl: true,
      minCd: true,
      minDclDcm: true
    },
    constraints: {
      minThickness: 0.08,
      maxThickness: 0.20,
      enforceManufacturability: true
    }
  });

  const styles = {
    header: {
      fontSize: '18px',
      fontWeight: '600',
      marginBottom: '16px',
      color: '#1976d2',
      borderBottom: '2px solid #e3f2fd',
      paddingBottom: '8px'
    },
    section: {
      marginBottom: '20px',
      padding: '12px',
      background: '#f8f9fa',
      borderRadius: '6px',
      border: '1px solid #e9ecef'
    },
    sectionTitle: {
      fontSize: '14px',
      fontWeight: '600',
      marginBottom: '12px',
      color: '#495057'
    },
    formGroup: {
      marginBottom: '12px'
    },
    label: {
      display: 'block',
      marginBottom: '4px',
      fontSize: '12px',
      fontWeight: '500',
      color: '#495057'
    },
    input: {
      width: '100%',
      padding: '8px',
      border: '1px solid #ced4da',
      borderRadius: '4px',
      fontSize: '13px',
      boxSizing: 'border-box'
    },
    select: {
      width: '100%',
      padding: '8px',
      border: '1px solid #ced4da',
      borderRadius: '4px',
      fontSize: '13px',
      backgroundColor: 'white'
    },
    checkboxContainer: {
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
      marginBottom: '6px'
    },
    checkbox: {
      width: '14px',
      height: '14px'
    },
    checkboxLabel: {
      fontSize: '12px',
      color: '#495057',
      cursor: 'pointer'
    },
    twoColumn: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gap: '8px'
    },
    progressContainer: {
      marginBottom: '16px'
    },
    progressBar: {
      width: '100%',
      height: '20px',
      backgroundColor: '#e9ecef',
      borderRadius: '10px',
      overflow: 'hidden',
      marginBottom: '8px'
    },
    progressFill: {
      height: '100%',
      backgroundColor: '#28a745',
      transition: 'width 0.3s ease',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      color: 'white',
      fontSize: '11px',
      fontWeight: '500'
    },
    progressText: {
      fontSize: '12px',
      color: '#6c757d',
      textAlign: 'center'
    },
    button: {
      width: '100%',
      padding: '12px',
      border: 'none',
      borderRadius: '6px',
      fontSize: '14px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.2s ease',
      marginTop: '8px'
    },
    primaryButton: {
      backgroundColor: '#28a745',
      color: 'white'
    },
    disabledButton: {
      backgroundColor: '#6c757d',
      color: '#adb5bd',
      cursor: 'not-allowed'
    },
    stopButton: {
      backgroundColor: '#dc3545',
      color: 'white'
    },
    algorithmInfo: {
      fontSize: '11px',
      color: '#6c757d',
      fontStyle: 'italic',
      marginTop: '4px'
    },
    objectiveWeights: {
      display: 'grid',
      gridTemplateColumns: 'auto 1fr auto',
      gap: '8px',
      alignItems: 'center',
      marginBottom: '8px'
    },
    weightSlider: {
      width: '100%',
      height: '4px',
      borderRadius: '2px',
      background: '#e9ecef',
      outline: 'none',
      WebkitAppearance: 'none',
      appearance: 'none'
    },
    weightValue: {
      fontSize: '11px',
      fontWeight: '500',
      color: '#495057',
      minWidth: '30px',
      textAlign: 'right'
    }
  };

  const handleConfigChange = (section, key, value) => {
    setConfig(prev => ({
      ...prev,
      [section]: {
        ...prev[section],
        [key]: value
      }
    }));
  };

  const handleOptimize = () => {
    const optimizationConfig = {
      algorithm: config.algorithm,
      population_size: config.populationSize,
      max_iterations: config.maxGenerations,
      objectives: Object.keys(config.objectives).filter(key => config.objectives[key]),
      constraints: config.constraints
    };

    onOptimize(optimizationConfig);
  };

  const algorithmOptions = [
    {
      value: 'nsga2',
      label: 'NSGA-II',
      description: 'Multi-objective genetic algorithm with fast non-dominated sorting'
    },
    {
      value: 'moead',
      label: 'MOEA/D',
      description: 'Decomposition-based multi-objective evolutionary algorithm'
    },
    {
      value: 'spea2',
      label: 'SPEA2',
      description: 'Strength Pareto Evolutionary Algorithm 2'
    }
  ];

  const selectedAlgorithm = algorithmOptions.find(opt => opt.value === config.algorithm);

  return (
    <div>
      <h2 style={styles.header}>Optimization Control</h2>

      {/* 진행 상황 */}
      {progress.isRunning && (
        <div style={styles.progressContainer}>
          <div style={styles.progressBar}>
            <div 
              style={{
                ...styles.progressFill,
                width: `${progress.progress}%`
              }}
            >
              {progress.progress.toFixed(0)}%
            </div>
          </div>
          <div style={styles.progressText}>
            Generation {progress.generation}: {progress.message}
          </div>
        </div>
      )}

      {/* 알고리즘 선택 */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>Algorithm Settings</div>
        
        <div style={styles.formGroup}>
          <label style={styles.label}>Optimization Algorithm</label>
          <select
            style={styles.select}
            value={config.algorithm}
            onChange={(e) => setConfig(prev => ({ ...prev, algorithm: e.target.value }))}
            disabled={progress.isRunning}
          >
            {algorithmOptions.map(opt => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <div style={styles.algorithmInfo}>
            {selectedAlgorithm?.description}
          </div>
        </div>

        <div style={styles.twoColumn}>
          <div style={styles.formGroup}>
            <label style={styles.label}>Population Size</label>
            <input
              type="number"
              style={styles.input}
              value={config.populationSize}
              min="10"
              max="100"
              step="10"
              onChange={(e) => setConfig(prev => ({ ...prev, populationSize: parseInt(e.target.value) || 20 }))}
              disabled={progress.isRunning}
            />
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>Max Generations</label>
            <input
              type="number"
              style={styles.input}
              value={config.maxGenerations}
              min="10"
              max="200"
              step="10"
              onChange={(e) => setConfig(prev => ({ ...prev, maxGenerations: parseInt(e.target.value) || 50 }))}
              disabled={progress.isRunning}
            />
          </div>
        </div>
      </div>

      {/* 목적함수 설정 */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>Objective Functions</div>
        
        <div style={styles.checkboxContainer}>
          <input
            type="checkbox"
            style={styles.checkbox}
            id="maxCl"
            checked={config.objectives.maxCl}
            onChange={(e) => handleConfigChange('objectives', 'maxCl', e.target.checked)}
            disabled={progress.isRunning}
          />
          <label style={styles.checkboxLabel} htmlFor="maxCl">
            Maximize Lift Coefficient (Cl)
          </label>
        </div>

        <div style={styles.checkboxContainer}>
          <input
            type="checkbox"
            style={styles.checkbox}
            id="minCd"
            checked={config.objectives.minCd}
            onChange={(e) => handleConfigChange('objectives', 'minCd', e.target.checked)}
            disabled={progress.isRunning}
          />
          <label style={styles.checkboxLabel} htmlFor="minCd">
            Minimize Drag Coefficient (Cd)
          </label>
        </div>

        <div style={styles.checkboxContainer}>
          <input
            type="checkbox"
            style={styles.checkbox}
            id="minDclDcm"
            checked={config.objectives.minDclDcm}
            onChange={(e) => handleConfigChange('objectives', 'minDclDcm', e.target.checked)}
            disabled={progress.isRunning}
          />
          <label style={styles.checkboxLabel} htmlFor="minDclDcm">
            Minimize dCl/dCm (Stability)
          </label>
        </div>
      </div>

      {/* 제약조건 설정 */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>Design Constraints</div>
        
        <div style={styles.twoColumn}>
          <div style={styles.formGroup}>
            <label style={styles.label}>Min Thickness (%c)</label>
            <input
              type="number"
              style={styles.input}
              value={(config.constraints.minThickness * 100).toFixed(1)}
              min="5"
              max="15"
              step="0.5"
              onChange={(e) => handleConfigChange('constraints', 'minThickness', parseFloat(e.target.value) / 100 || 0.08)}
              disabled={progress.isRunning}
            />
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>Max Thickness (%c)</label>
            <input
              type="number"
              style={styles.input}
              value={(config.constraints.maxThickness * 100).toFixed(1)}
              min="15"
              max="30"
              step="0.5"
              onChange={(e) => handleConfigChange('constraints', 'maxThickness', parseFloat(e.target.value) / 100 || 0.20)}
              disabled={progress.isRunning}
            />
          </div>
        </div>

        <div style={styles.checkboxContainer}>
          <input
            type="checkbox"
            style={styles.checkbox}
            id="manufacturability"
            checked={config.constraints.enforceManufacturability}
            onChange={(e) => handleConfigChange('constraints', 'enforceManufacturability', e.target.checked)}
            disabled={progress.isRunning}
          />
          <label style={styles.checkboxLabel} htmlFor="manufacturability">
            Enforce Manufacturability Constraints
          </label>
        </div>
      </div>

      {/* 최적화 버튼 */}
      <button
        style={{
          ...styles.button,
          ...(progress.isRunning ? styles.stopButton : styles.primaryButton),
          ...(disabled && !progress.isRunning ? styles.disabledButton : {})
        }}
        onClick={progress.isRunning ? () => {} : handleOptimize}
        disabled={disabled && !progress.isRunning}
        onMouseEnter={(e) => {
          if (!disabled && !progress.isRunning) {
            e.target.style.backgroundColor = '#218838';
          }
        }}
        onMouseLeave={(e) => {
          if (!disabled && !progress.isRunning) {
            e.target.style.backgroundColor = '#28a745';
          }
        }}
      >
        {progress.isRunning ? 
          '⏹️ Stop Optimization' : 
          '🚀 Start Optimization'
        }
      </button>

      {/* 예상 시간 정보 */}
      {!progress.isRunning && (
        <div style={{ ...styles.algorithmInfo, textAlign: 'center', marginTop: '8px' }}>
          Estimated time: ~{Math.ceil(config.populationSize * config.maxGenerations / 100)} minutes
          <br />
          ({config.populationSize} individuals × {config.maxGenerations} generations)
        </div>
      )}
    </div>
  );
};

export default OptimizationPanel;