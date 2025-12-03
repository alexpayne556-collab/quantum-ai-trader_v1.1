import React from 'react';
import { Card, CardContent, Typography, Box, LinearProgress } from '@mui/material';

const QuantScoreCard = ({ symbol, quantScore, factors, statisticalSignificance, regime }) => {
  const getScoreColor = (score) => {
    if (score >= 80) return '#00ff88';
    if (score >= 60) return '#ffa500';
    return '#ff4444';
  };

  const getRegimeColor = (regime) => {
    switch (regime) {
      case 'bull': return '#00ff88';
      case 'bear': return '#ff4444';
      case 'sideways': return '#ffa500';
      default: return '#ffffff';
    }
  };

  return (
    <Card sx={{ 
      background: 'linear-gradient(135deg, #1e3a5f 0%, #2d4a6f 100%)',
      border: `3px solid ${getScoreColor(quantScore)}`,
      borderRadius: '20px',
      padding: '24px',
      textAlign: 'center',
      boxShadow: '0 12px 24px rgba(0,0,0,0.4)'
    }}>
      <CardContent>
        <Typography variant="h6" sx={{ color: '#a0a0b0', marginBottom: '16px' }}>
          {symbol} Quant Score
        </Typography>
        
        <Typography 
          variant="h1" 
          sx={{ 
            fontSize: '72px',
            fontWeight: 900,
            background: `linear-gradient(135deg, ${getScoreColor(quantScore)} 0%, ${getScoreColor(quantScore)}dd 100%)`,
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            fontFamily: 'Roboto Mono, monospace',
            lineHeight: 1,
            marginBottom: '8px'
          }}
        >
          {quantScore?.toFixed(1) || 'N/A'}
        </Typography>
        
        <Typography 
          variant="body2" 
          sx={{ 
            color: '#a0a0b0',
            textTransform: 'uppercase',
            letterSpacing: '2px',
            marginBottom: '16px'
          }}
        >
          Statistical Significance: {(statisticalSignificance * 100)?.toFixed(1)}%
        </Typography>
        
        <Box sx={{ marginBottom: '16px' }}>
          <Typography 
            variant="body1" 
            sx={{ 
              color: getRegimeColor(regime),
              fontWeight: 'bold',
              textTransform: 'uppercase',
              letterSpacing: '1px'
            }}
          >
            Regime: {regime || 'Unknown'}
          </Typography>
        </Box>
        
        {factors && (
          <Box sx={{ marginTop: '24px' }}>
            <Typography variant="h6" sx={{ color: '#ffffff', marginBottom: '12px' }}>
              Factor Breakdown
            </Typography>
            {Object.entries(factors).map(([key, value]) => (
              <Box key={key} sx={{ marginBottom: '8px' }}>
                <Typography variant="body2" sx={{ color: '#a0a0b0', textTransform: 'capitalize' }}>
                  {key.replace('_', ' ')}: {value}
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={value} 
                  sx={{ 
                    height: '4px',
                    borderRadius: '2px',
                    backgroundColor: '#2a2a3a',
                    '& .MuiLinearProgress-bar': {
                      backgroundColor: getScoreColor(value)
                    }
                  }}
                />
              </Box>
            ))}
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default QuantScoreCard;
