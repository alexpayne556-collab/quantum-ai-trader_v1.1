import React, { useState, useEffect } from 'react';
import { 
  Grid, 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  Button,
  TextField,
  Alert,
  CircularProgress
} from '@mui/material';
import QuantScoreCard from './QuantScoreCard';
import { fetchQuantScore, fetchBayesianSignal, fetchRegimeDetection } from '../services/api';

const InstitutionalDashboard = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [data, setData] = useState(null);

  const analyzeSymbol = async () => {
    if (!symbol) {
      setError('Please enter a symbol');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const [quantScore, bayesianSignal, regime] = await Promise.all([
        fetchQuantScore(symbol),
        fetchBayesianSignal(symbol),
        fetchRegimeDetection(symbol)
      ]);

      setData({
        symbol,
        quantScore,
        bayesianSignal,
        regime
      });
    } catch (err) {
      setError(`Failed to analyze ${symbol}: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box sx={{ padding: '24px', backgroundColor: '#0a0e27', minHeight: '100vh' }}>
      <Typography 
        variant="h3" 
        sx={{ 
          color: '#ffffff', 
          marginBottom: '32px',
          textAlign: 'center',
          fontWeight: 'bold'
        }}
      >
        🏆 Quantum AI Institutional Dashboard
      </Typography>

      {/* Input Section */}
      <Card sx={{ marginBottom: '32px', backgroundColor: '#1a1f3a' }}>
        <CardContent>
          <Box sx={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            <TextField
              label="Stock Symbol"
              variant="outlined"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              sx={{ 
                backgroundColor: '#2a2f4a',
                inputProps: { style: { color: '#ffffff' } },
                label: { style: { color: '#a0a0b0' } }
              }}
            />
            <Button 
              variant="contained" 
              onClick={analyzeSymbol}
              disabled={loading}
              sx={{ 
                backgroundColor: '#00ff88',
                color: '#000000',
                fontWeight: 'bold',
                '&:hover': { backgroundColor: '#00d86a' }
              }}
            >
              {loading ? <CircularProgress size={24} /> : 'Analyze'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {error && (
        <Alert severity="error" sx={{ marginBottom: '24px' }}>
          {error}
        </Alert>
      )}

      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', padding: '48px' }}>
          <CircularProgress size={48} />
        </Box>
      )}

      {data && !loading && (
        <Grid container spacing={3}>
          {/* Quant Score Card */}
          <Grid item xs={12} md={4}>
            <QuantScoreCard
              symbol={data.symbol}
              quantScore={data.quantScore?.quant_score}
              factors={data.quantScore?.factors}
              statisticalSignificance={data.quantScore?.statistical_significance}
              regime={data.regime?.regime}
            />
          </Grid>

          {/* Bayesian Signal Card */}
          <Grid item xs={12} md={4}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #1a1f3a 0%, #242942 100%)',
              borderLeft: '4px solid #4b7bec',
              boxShadow: '0 4px 8px rgba(0,0,0,0.3)'
            }}>
              <CardContent>
                <Typography variant="h6" sx={{ color: '#ffffff', marginBottom: '16px' }}>
                  Bayesian Signal Fusion
                </Typography>
                <Typography 
                  variant="h4" 
                  sx={{ 
                    color: data.bayesianSignal?.signal === 'BUY' ? '#00ff88' : '#ff4444',
                    fontWeight: 'bold',
                    marginBottom: '8px'
                  }}
                >
                  {data.bayesianSignal?.signal || 'N/A'}
                </Typography>
                <Typography variant="body2" sx={{ color: '#a0a0b0' }}>
                  Confidence: {((data.bayesianSignal?.confidence || 0) * 100)?.toFixed(1)}%
                </Typography>
                <Typography variant="body2" sx={{ color: '#a0a0b0', marginTop: '8px' }}>
                  Expected Return: {data.bayesianSignal?.expected_return?.toFixed(2)}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Regime Detection Card */}
          <Grid item xs={12} md={4}>
            <Card sx={{ 
              background: 'linear-gradient(135deg, #1a1f3a 0%, #242942 100%)',
              borderLeft: '4px solid #ffa500',
              boxShadow: '0 4px 8px rgba(0,0,0,0.3)'
            }}>
              <CardContent>
                <Typography variant="h6" sx={{ color: '#ffffff', marginBottom: '16px' }}>
                  Market Regime Detection
                </Typography>
                <Typography 
                  variant="h4" 
                  sx={{ 
                    color: data.regime?.regime === 'bull' ? '#00ff88' : 
                           data.regime?.regime === 'bear' ? '#ff4444' : '#ffa500',
                    fontWeight: 'bold',
                    textTransform: 'uppercase',
                    marginBottom: '8px'
                  }}
                >
                  {data.regime?.regime || 'Unknown'}
                </Typography>
                <Typography variant="body2" sx={{ color: '#a0a0b0' }}>
                  Confidence: {((data.regime?.confidence || 0) * 100)?.toFixed(1)}%
                </Typography>
                <Typography variant="body2" sx={{ color: '#a0a0b0', marginTop: '8px' }}>
                  Duration: {data.regime?.duration_days || 0} days
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Additional Metrics */}
          <Grid item xs={12}>
            <Card sx={{ backgroundColor: '#1a1f3a' }}>
              <CardContent>
                <Typography variant="h6" sx={{ color: '#ffffff', marginBottom: '16px' }}>
                  Institutional Metrics
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6} md={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" sx={{ color: '#00ff88', fontWeight: 'bold' }}>
                        {data.quantScore?.kelly_size ? (data.quantScore.kelly_size * 100)?.toFixed(1) : 'N/A'}%
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#a0a0b0' }}>
                        Kelly Position Size
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" sx={{ color: '#4b7bec', fontWeight: 'bold' }}>
                        {data.bayesianSignal?.sharpe_ratio?.toFixed(2) || 'N/A'}
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#a0a0b0' }}>
                        Sharpe Ratio
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" sx={{ color: '#ffa500', fontWeight: 'bold' }}>
                        {data.quantScore?.max_drawdown ? (data.quantScore.max_drawdown * 100)?.toFixed(1) : 'N/A'}%
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#a0a0b0' }}>
                        Max Drawdown
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={6} md={3}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" sx={{ color: '#ff6b6b', fontWeight: 'bold' }}>
                        {data.bayesianSignal?.win_rate ? (data.bayesianSignal.win_rate * 100)?.toFixed(1) : 'N/A'}%
                      </Typography>
                      <Typography variant="body2" sx={{ color: '#a0a0b0' }}>
                        Win Rate
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default InstitutionalDashboard;
