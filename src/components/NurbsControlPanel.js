import React, { useState, useCallback } from 'react';

const NurbsControlPanel = ({ 
  controlParameters, 
  parameterBounds, 
  onParametersChange, 
  disabled 
}) => {
  const [presetMode, setPresetMode] = useState(false);

  const styles = {
    header: {
      fontSize: '18px',
      fontWeight: '600',
      marginBottom: '20px',
      color: '#1976d2',
      borderBottom: '2px solid #e3f2fd',
      paddingBottom: '8px'
    },
    modeToggle: {
      display: 'flex',
      marginBottom: '16px',
      background: '#f5f5f5',
      borderRadius: '6px',
      padding: '2px'
    },
    modeButton: {
      flex: 1,
      padding: '8px 16px',
      border: 'none',
      background: 'transparent',
      cursor: 'pointer',
      borderRadius: '4px',
      fontSize: '12px',
      fontWeight: '500',
      transition: 'all 0.2s ease'
    },
    modeButtonActive: {
      background: 'white',
      boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
      color: '#1976d2'
    },
    controlGrid: {
      maxHeight: '300px',
      overflowY: 'auto',
      border: '1px solid #e0e0e0',
      borderRadius: '6px',
      padding: '12px'
    },
    controlItem: {
      marginBottom: '12px',
      padding: '8px',
      background: '#fafafa',
      borderRadius: '4px'
    },
    controlLabel: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '6px',
      fontSize: '12px'
    },
    controlName: {
      fontWeight: '500',
      color: '#424242'
    },
    controlValue: {
      fontSize: '11px',
      color: '#1976d2',
      fontWeight: '600',
      fontFamily: 'monospace'
    },
    slider: {
      width: '100%',
      height: '4px',
      borderRadius: '2px',
      background: '#e0e0e0',
      outline: 'none',
      cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? 0.5 : 1,
      WebkitAppearance: 'none',
      appearance: 'none'
    },
    sliderThumb: {
      width: '16px',
      height: '16px',
      borderRadius: '50%',
      background: '#1976d2',
      cursor: disabled ? 'not-allowed' : 'pointer',
      border: '2px solid white',
      boxShadow: '0 1px 3px rgba(0,0,0,0.3)'
    },
    presetGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(2, 1fr)',
      gap: '8px',
      marginBottom: '16px'
    },
    presetButton: {
      padding: '8px 12px',
      border: '1px solid #e0e0e0',
      borderRadius: '4px',
      background: 'white',
      cursor: 'pointer',
      fontSize: '11px',
      fontWeight: '500',
      transition: 'all 0.2s ease'
    },
    presetButtonHover: {
      borderColor: '#1976d2',
      color: '#1976d2'
    },
    buttonGroup: {
      display: 'flex',
      gap: '8px',
      marginTop: '16px'
    },
    button: {
      flex: 1,
      padding: '10px 16px',
      border: 'none',
      borderRadius: '4px',
      fontSize: '12px',
      fontWeight: '500',
      cursor: 'pointer',
      transition: 'all 0.2s ease'
    },
    primaryButton: {
      background: '#1976d2',
      color: 'white'
    },
    secondaryButton: {
      background: '#f5f5f5',
      color: '#666',
      border: '1px solid #e0e0e0'
    },
    disabledButton: {
      opacity: 0.5,
      cursor: 'not-allowed'
    },
    infoBox: {
      background: '#e3f2fd',
      border: '1px solid #bbdefb',
      borderRadius: '4px',
      padding: '8px',
      fontSize: '11px',
      color: '#1565c0',
      marginTop: '12px'
    },
    stats: {
      background: '#f8f9fa',
      border: '1px solid #dee2e6',
      borderRadius: '4px',
      padding: '8px',
      fontSize: '11px',
      marginBottom: '12px'
    },
    statsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(2, 1fr)',
      gap: '8px'
    },
    statItem: {
      display: 'flex',
      justifyContent: 'space-between'
    }
  };

  // NURBS 제어점 프리셋
  const airfoilPresets = {
    'NACA 0012': [0.002, 0.008, 0.012, 0.010, 0.004, 0.000, -0.002, -0.008, -0.012, -0.010, -0.004, 0.000],
    'NACA 2412': [0.004, 0.012, 0.016, 0.014, 0.008, 0.002, -0.000, -0.006, -0.010, -0.008, -0.002, 0.002],
    'NACA 4412': [0.008, 0.016, 0.020, 0.018, 0.012, 0.004, 0.002, -0.004, -0.008, -0.006, 0.000, 0.004],
    'Symmetric': [0.000, 0.008, 0.012, 0.010, 0.004, 0.000, 0.000, -0.008, -0.012, -0.010, -0.004, 0.000],
    'Flat Plate': [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    'High Camber': [0.010, 0.020, 0.024, 0.022, 0.016, 0.008, 0.004, -0.002, -0.006, -0.004, 0.002, 0.008]
  };

  const handleParameterChange = useCallback((index, value) => {
    if (disabled) return;
    
    const newParameters = [...controlParameters];
    newParameters[index] = parseFloat(value);
    onParametersChange(newParameters);
  }, [controlParameters, onParametersChange, disabled]);

  const handlePresetApply = useCallback((presetName) => {
    if (disabled) return;
    
    const presetValues = airfoilPresets[presetName];
    if (presetValues && presetValues.length <= controlParameters.length) {
      const newParameters = [...controlParameters];
      presetValues.forEach((value, index) => {
        if (index < newParameters.length) {
          newParameters[index] = value;
        }
      });
      onParametersChange(newParameters);
    }
  }, [controlParameters, onParametersChange, disabled]);

  const handleReset = useCallback(() => {
    if (disabled) return;
    
    const resetParameters = new Array(controlParameters.length).fill(0);
    onParametersChange(resetParameters);
  }, [controlParameters.length, onParametersChange, disabled]);

  const handleRandomize = useCallback(() => {
    if (disabled) return;
    
    const randomParameters = controlParameters.map((_, index) => {
      const min = parameterBounds.lower?.[index] || -0.05;
      const max = parameterBounds.upper?.[index] || 0.05;
      return Math.random() * (max - min) + min;
    });
    onParametersChange(randomParameters);
  }, [controlParameters, parameterBounds, onParametersChange, disabled]);

  // 통계 계산
  const calculateStats = () => {
    if (controlParameters.length === 0) return null;
    
    const nonZero = controlParameters.filter(p => Math.abs(p) > 0.001);
    const maxDeformation = Math.max(...controlParameters.map(Math.abs));
    const avgDeformation = controlParameters.reduce((sum, p) => sum + Math.abs(p), 0) / controlParameters.length;
    
    return {
      activePoints: nonZero.length,
      totalPoints: controlParameters.length,
      maxDeformation: maxDeformation.toFixed(4),
      avgDeformation: avgDeformation.toFixed(4)
    };
  };

  const stats = calculateStats();

  return (
    <div>
      <h2 style={styles.header}>NURBS Control Points</h2>

      {/* 통계 정보 */}
      {stats && (
        <div style={styles.stats}>
          <div style={styles.statsGrid}>
            <div style={styles.statItem}>
              <span>Active Points:</span>
              <span>{stats.activePoints}/{stats.totalPoints}</span>
            </div>
            <div style={styles.statItem}>
              <span>Max Deformation:</span>
              <span>{stats.maxDeformation}</span>
            </div>
            <div style={styles.statItem}>
              <span>Avg Deformation:</span>
              <span>{stats.avgDeformation}</span>
            </div>
            <div style={styles.statItem}>
              <span>Range:</span>
              <span>±{((parameterBounds.upper?.[0] || 0.05) * 100).toFixed(1)}%c</span>
            </div>
          </div>
        </div>
      )}

      {/* 모드 전환 */}
      <div style={styles.modeToggle}>
        <button
          style={{
            ...styles.modeButton,
            ...(!presetMode ? styles.modeButtonActive : {})
          }}
          onClick={() => setPresetMode(false)}
        >
          Manual Control
        </button>
        <button
          style={{
            ...styles.modeButton,
            ...(presetMode ? styles.modeButtonActive : {})
          }}
          onClick={() => setPresetMode(true)}
        >
          Presets
        </button>
      </div>

      {/* 프리셋 모드 */}
      {presetMode ? (
        <div>
          <div style={styles.presetGrid}>
            {Object.keys(airfoilPresets).map(presetName => (
              <button
                key={presetName}
                style={styles.presetButton}
                onClick={() => handlePresetApply(presetName)}
                disabled={disabled}
                onMouseEnter={(e) => {
                  if (!disabled) {
                    Object.assign(e.target.style, styles.presetButtonHover);
                  }
                }}
                onMouseLeave={(e) => {
                  if (!disabled) {
                    e.target.style.borderColor = '#e0e0e0';
                    e.target.style.color = 'inherit';
                  }
                }}
              >
                {presetName}
              </button>
            ))}
          </div>
        </div>
      ) : (
        /* 수동 제어 모드 */
        <div style={styles.controlGrid}>
          {controlParameters.map((value, index) => {
            const min = parameterBounds.lower?.[index] || -0.05;
            const max = parameterBounds.upper?.[index] || 0.05;
            const isUpper = index < Math.floor(controlParameters.length / 2);
            const pointIndex = isUpper ? index + 1 : index - Math.floor(controlParameters.length / 2) + 1;
            const surface = isUpper ? 'Upper' : 'Lower';
            
            return (
              <div key={index} style={styles.controlItem}>
                <div style={styles.controlLabel}>
                  <span style={styles.controlName}>
                    {surface} CP{pointIndex}
                  </span>
                  <span style={styles.controlValue}>
                    {value.toFixed(4)}
                  </span>
                </div>
                <input
                  type="range"
                  style={styles.slider}
                  min={min}
                  max={max}
                  step="0.001"
                  value={value}
                  onChange={(e) => handleParameterChange(index, e.target.value)}
                  disabled={disabled}
                />
              </div>
            );
          })}
        </div>
      )}

      {/* 버튼 그룹 */}
      <div style={styles.buttonGroup}>
        <button
          style={{
            ...styles.button,
            ...styles.secondaryButton,
            ...(disabled ? styles.disabledButton : {})
          }}
          onClick={handleReset}
          disabled={disabled}
        >
          🔄 Reset
        </button>
        <button
          style={{
            ...styles.button,
            ...styles.secondaryButton,
            ...(disabled ? styles.disabledButton : {})
          }}
          onClick={handleRandomize}
          disabled={disabled}
        >
          🎲 Randomize
        </button>
      </div>

      {/* 정보 박스 */}
      <div style={styles.infoBox}>
        <strong>NURBS Control:</strong><br />
        • Each slider controls a NURBS control point<br />
        • Values represent y-displacement as % of chord<br />
        • Leading/Trailing edges are fixed<br />
        • Upper and lower surfaces controlled independently
      </div>
    </div>
  );
};

export default NurbsControlPanel;