"""
Advanced AI Recommender
- Walk-forward validation (rolling windows)
- HistGradientBoostingClassifier (sklearn) with calibration (CalibratedClassifierCV)
- Save final calibrated model per ticker to `models/<ticker>_model.pkl`
- Integrate with `NotificationService` to send alerts when confidence >= threshold

Run: python ai_recommender_adv.py
"""
import os
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Import notification service
from notification_service import NotificationService, Alert

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Reuse simple feature engineering and labeling
class FE:
    @staticmethod
    def get_array(df, col):
        if isinstance(df[col], pd.DataFrame):
            return df[col].iloc[:, 0].values
        return df[col].values

    @staticmethod
    def engineer(df: pd.DataFrame) -> pd.DataFrame:
        close = np.asarray(FE.get_array(df, 'Close'), dtype='float64')
        high = np.asarray(FE.get_array(df, 'High'), dtype='float64')
        low = np.asarray(FE.get_array(df, 'Low'), dtype='float64')
        volume = np.asarray(FE.get_array(df, 'Volume'), dtype='float64')

        out = pd.DataFrame(index=df.index)
        out['rsi_9'] = talib.RSI(close, timeperiod=9)
        out['rsi_14'] = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=5, slowperiod=13, signalperiod=1)
        out['macd'] = macd
        out['macd_signal'] = macd_signal
        out['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
        out['adx'] = talib.ADX(high, low, close, timeperiod=14)
        out['ema_5'] = talib.EMA(close, timeperiod=5)
        out['ema_13'] = talib.EMA(close, timeperiod=13)
        out['ema_diff'] = out['ema_5'] - out['ema_13']
        out['vol_sma_20'] = talib.SMA(volume, timeperiod=20)
        out['vol_ratio'] = volume / (out['vol_sma_20'] + 1e-9)
        out['returns_1'] = pd.Series(close).pct_change(1)
        out['returns_5'] = pd.Series(close).pct_change(5)
        out['obv'] = talib.OBV(close, volume)

        # cleanup
        out = out.replace([np.inf, -np.inf], np.nan)
        out = out.bfill().ffill()  # Pandas 2.0+ compatible
        for c in out.columns:
            if out[c].isna().any():
                med = out[c].median()
                if pd.isna(med):
                    med = 0.0
                out[c] = out[c].fillna(med)
        return out

class Label:
    @staticmethod
    def make(df: pd.DataFrame, horizon=7, thr=0.02):
        close = FE.get_array(df, 'Close')
        s = pd.Series(close, index=df.index)
        fut = (s.shift(-horizon) - s) / s
        labels = pd.Series(1, index=df.index)
        labels.loc[fut > thr] = 2
        labels.loc[fut < -thr] = 0
        return labels


def walk_forward_train(df, horizon=7, initial_train_days=365*2, test_window_days=90):
    """Perform walk-forward validation and return aggregated metrics and final trained model"""
    features = FE.engineer(df)
    labels = Label.make(df, horizon=horizon)

    # cut the future horizon
    features = features.iloc[:-horizon]
    labels = labels.iloc[:-horizon]

    dates = features.index
    start = dates[0]
    end = dates[-1]

    windows = []
    train_start = start
    train_end = start + pd.Timedelta(days=initial_train_days)
    # ensure min
    if train_end >= end:
        raise ValueError('Not enough data for initial training window')

    metrics = []
    final_model = None
    scaler = StandardScaler()

    while train_end < end:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_window_days)
        if test_start >= end:
            break
        if test_end > end:
            test_end = end
        # slice
        X_train = features.loc[train_start:train_end]
        y_train = labels.loc[train_start:train_end]
        X_test = features.loc[test_start:test_end]
        y_test = labels.loc[test_start:test_end]

        # fit model
        try:
            Xtr = scaler.fit_transform(X_train)
            Xte = scaler.transform(X_test)
            base = HistGradientBoostingClassifier(random_state=42)
            # calibrate using internal CV on train
            calib = CalibratedClassifierCV(base, method='sigmoid', cv=3)
            calib.fit(Xtr, y_train.values)
            y_pred = calib.predict(Xte)
            y_prob = calib.predict_proba(Xte)
            acc = accuracy_score(y_test, y_pred)
            metrics.append({'train_start': train_start, 'train_end': train_end, 'test_start': test_start, 'test_end': test_end, 'accuracy': acc})
            final_model = calib
        except Exception as e:
            print('Train failed on window:', e)

        # advance window
        train_end = train_end + pd.Timedelta(days=test_window_days)

    return metrics, final_model, scaler, features, labels


def train_and_save(ticker: str):
    print('===== Training', ticker, '====')
    df = yf.download(ticker, period='5y', progress=False, auto_adjust=True)
    if len(df) < 800:
        print('Insufficient rows for robust walk-forward; continuing with available data')
    try:
        metrics, model, scaler, features, labels = walk_forward_train(df)
    except Exception as e:
        print('Walk-forward failed:', e)
        return {'error': str(e)}

    if model is None:
        return {'error': 'model_training_failed'}

    # Fit final model on all available data
    X = features.loc[:]
    y = labels.loc[:]
    Xs = scaler.fit_transform(X)

    # Retrain final calibrated model on all data
    base_final = HistGradientBoostingClassifier(random_state=42)
    calib_final = CalibratedClassifierCV(base_final, method='sigmoid', cv=3)
    try:
        calib_final.fit(Xs, y.values)
    except Exception as e:
        print('Final model training failed:', e)
        return {'error': 'final_training_failed', 'exception': str(e)}

    model_path = os.path.join(MODEL_DIR, f'{ticker}_model.pkl')
    joblib.dump({'model': calib_final, 'scaler': scaler}, model_path)
    print('Saved model to', model_path)

    return {'ticker': ticker, 'metrics': metrics, 'model_path': model_path}


async def send_alert_if_confident(ticker: str, model_path: str, threshold: float = 0.65):
    # Load model
    data = joblib.load(model_path)
    model = data['model']
    scaler = data['scaler']

    # fetch latest
    df = yf.download(ticker, period='1y', progress=False, auto_adjust=True)
    features = FE.engineer(df)
    X_latest = features.iloc[-1:]
    Xs = scaler.transform(X_latest)
    probs = model.predict_proba(Xs)[0]
    classes = model.classes_
    best_idx = int(np.argmax(probs))
    best_class = int(classes[best_idx])
    label_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    signal = label_map.get(best_class, 'HOLD')
    conf = float(probs[best_idx])

    print(f'{ticker} predicted {signal} with confidence {conf:.3f}')

    if conf >= threshold:
        # Build alert - limited fields available, fill with defaults
        price = float(FE.get_array(df, 'Close')[-1])
        now = datetime.now()
        alert = Alert(
            ticker=ticker,
            signal=signal,
            price=price,
            signal_strength=conf * 100,
            rsi_5m=0.0,
            rsi_1h=0.0,
            macd_signal='N/A',
            momentum=0.0,
            timestamp=now,
            priority='MEDIUM',
            channel=['discord']
        )
        ns = NotificationService()
        await ns.send_alert(alert)
        return {'sent': True, 'signal': signal, 'confidence': conf}
    return {'sent': False, 'signal': signal, 'confidence': conf}


if __name__ == '__main__':
    import asyncio
    tickers = ['MU', 'APLD', 'IONQ', 'ANNX']
    results = {}
    for t in tickers:
        res = train_and_save(t)
        print('Train result:', res)
        results[t] = res

    # attempt alerts
    async def run_alerts():
        for t, r in results.items():
            if isinstance(r, dict) and r.get('model_path'):
                out = await send_alert_if_confident(t, r['model_path'], threshold=0.65)
                print('Alert:', t, out)
    asyncio.run(run_alerts())
