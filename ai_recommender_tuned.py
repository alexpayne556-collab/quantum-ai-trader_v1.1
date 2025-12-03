"""
Tuned AI Recommender
- Randomized hyperparameter search for HistGradientBoostingClassifier
- Optional LightGBM (if installed) included in ensemble
- Soft-voting ensemble + probability calibration (CalibratedClassifierCV)
- Compute Brier score and save calibration data
- Send alerts via NotificationService when calibrated confidence >= ALERT_THRESHOLD (env)

Run: python ai_recommender_tuned.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import talib
from datetime import datetime, timedelta
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, accuracy_score
from scipy.stats import randint, uniform

# Optional LightGBM
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

from notification_service import NotificationService, Alert

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# feature engineering (reused but compact)
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
        # Use standard MACD parameters (12, 26, 9) - more robust, requires less data
        macd, macd_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        out['macd'] = macd
        out['macd_sig'] = macd_signal
        out['atr_14'] = talib.ATR(high, low, close, timeperiod=14)
        out['adx'] = talib.ADX(high, low, close, timeperiod=14)
        out['ema5'] = talib.EMA(close, timeperiod=5)
        out['ema13'] = talib.EMA(close, timeperiod=13)
        out['ema_diff'] = out['ema5'] - out['ema13']
        out['vol_sma20'] = talib.SMA(volume, timeperiod=20)
        out['vol_ratio'] = volume / (out['vol_sma20'] + 1e-9)
        out['ret1'] = pd.Series(close).pct_change(1)
        out['ret5'] = pd.Series(close).pct_change(5)
        out['obv'] = talib.OBV(close, volume)

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
    def make(df, horizon=7, thr=0.02):
        close = FE.get_array(df, 'Close')
        s = pd.Series(close, index=df.index)
        fut = (s.shift(-horizon) - s) / s
        labels = pd.Series(1, index=df.index)
        labels.loc[fut > thr] = 2
        labels.loc[fut < -thr] = 0
        return labels


def search_and_train(X_train, y_train, random_state=42):
    # Define HistGB hyperparam distribution
    param_dist = {
        'learning_rate': uniform(0.01, 0.3),
        'max_iter': randint(50, 400),
        'max_depth': randint(1, 16),
        'min_samples_leaf': randint(5, 100),
        'l2_regularization': uniform(0.0, 1.0)
    }
    base = HistGradientBoostingClassifier(random_state=random_state)
    rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=8, scoring='accuracy', cv=3, random_state=random_state, n_jobs=1)
    rs.fit(X_train, y_train)
    best = rs.best_estimator_
    return best, rs.best_params_, rs.best_score_


def build_ensemble(X_train, y_train):
    estimators = []
    # HistGB tuned
    try:
        hgb, params, score = search_and_train(X_train, y_train)
        estimators.append(('hgb', hgb))
    except Exception as e:
        print('HistGB search failed:', e)

    # LightGBM fallback
    if LGB_AVAILABLE:
        try:
            lgbm = lgb.LGBMClassifier(n_estimators=200)
            lgbm.fit(X_train, y_train)
            estimators.append(('lgb', lgbm))
        except Exception as e:
            print('LightGBM failed:', e)

    # Simple logistic baseline
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)
    estimators.append(('lr', lr))

    # Build voting
    if len(estimators) == 1:
        return estimators[0][1]
    vc = VotingClassifier(estimators=estimators, voting='soft')
    vc.fit(X_train, y_train)
    return vc


def train_ticker(ticker, horizon=7, test_size=0.2):
    print('--- Train', ticker)
    df = yf.download(ticker, period='5y', progress=False, auto_adjust=True)
    if len(df) < 250:
        print('Warning: limited data for', ticker)
    X = FE.engineer(df)
    y = Label.make(df, horizon=horizon)
    X = X.iloc[:-horizon]
    y = y.iloc[:-horizon]

    # drop remaining NaNs
    valid = X.dropna().index
    X = X.loc[valid]
    y = y.loc[valid]

    # train/test split (time series aware: last portion for test)
    split_idx = int(len(X) * (1 - test_size))
    X_tr, X_te = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr, y_te = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # build ensemble
    ensemble = build_ensemble(X_tr_s, y_tr)

    # calibrate
    try:
        calib = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
        calib.fit(X_tr_s, y_tr)
    except Exception:
        calib = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
        calib.fit(X_tr_s, y_tr)

    # evaluate
    y_pred = calib.predict(X_te_s)
    y_prob = calib.predict_proba(X_te_s)
    acc = accuracy_score(y_te, y_pred)
    brier = brier_score_loss((y_te == 2).astype(int), y_prob[:, list(calib.classes_).index(2)])

    # save model + scaler + meta
    model_path = os.path.join(MODEL_DIR, f'{ticker}_tuned.pkl')
    joblib.dump({'model': calib, 'scaler': scaler, 'classes': calib.classes_}, model_path)

    # save calibration info
    calib_info = {
        'ticker': ticker,
        'accuracy': float(acc),
        'brier_buy': float(brier),
        'n_train': int(len(X_tr)),
        'n_test': int(len(X_te)),
        'model_path': model_path
    }
    with open(os.path.join(MODEL_DIR, f'{ticker}_calibration.json'), 'w') as f:
        json.dump(calib_info, f, indent=2)

    print('Trained', ticker, 'acc', acc, 'brier_buy', brier)
    return calib_info


async def maybe_alert(ticker, model_meta, threshold: float = None):
    if threshold is None:
        try:
            threshold = float(os.getenv('ALERT_THRESHOLD', '0.65'))
        except Exception:
            threshold = 0.65
    path = model_meta['model_path']
    data = joblib.load(path)
    model = data['model']
    scaler = data['scaler']

    df = yf.download(ticker, period='1y', progress=False, auto_adjust=True)
    X = FE.engineer(df)
    X_latest = X.iloc[-1:]
    Xs = scaler.transform(X_latest)
    probs = model.predict_proba(Xs)[0]
    classes = model.classes_
    best_idx = int(np.argmax(probs))
    best_class = int(classes[best_idx])
    label_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    sig = label_map.get(best_class, 'HOLD')
    conf = float(probs[best_idx])
    print(f'{ticker} tuned predict: {sig} conf={conf:.3f}')

    if conf >= threshold:
        price = float(FE.get_array(df, 'Close')[-1])
        alert = Alert(
            ticker=ticker,
            signal=sig,
            price=price,
            signal_strength=conf*100,
            rsi_5m=0.0,
            rsi_1h=0.0,
            macd_signal='N/A',
            momentum=0.0,
            timestamp=datetime.now(),
            priority='MEDIUM',
            channel=['discord']
        )
        ns = NotificationService()
        await ns.send_alert(alert)
        return {'sent': True, 'signal': sig, 'confidence': conf}
    return {'sent': False, 'signal': sig, 'confidence': conf}


if __name__ == '__main__':
    import asyncio
    tickers = ['MU', 'APLD', 'IONQ', 'ANNX']
    metas = {}
    for t in tickers:
        try:
            m = train_ticker(t)
            metas[t] = m
        except Exception as e:
            print('Train error for', t, e)

    async def run_alerts():
        for t,m in metas.items():
            out = await maybe_alert(t, m)
            print('Alert attempt:', t, out)
    asyncio.run(run_alerts())
