from pathlib import Path
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'AAPL_model_dataset.csv'
OUTPUT = ROOT / 'data' / 'li_lstm_results.csv'
TRAIN_END = pd.Timestamp('2015-06-09')
VAL_END = pd.Timestamp('2015-09-17')
TEST_START = pd.Timestamp('2015-09-22')

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception: pass

def add_technical_features(df):
    d = df.copy()
    d['movement_lag1'] = d['MovementPercent'].shift(1)
    d['movement_mean_3'] = d['MovementPercent'].rolling(3).mean()
    d['movement_mean_7'] = d['MovementPercent'].rolling(7).mean()
    d['movement_std_7'] = d['MovementPercent'].rolling(7).std()
    d['volume_change'] = d['Volume'].pct_change()
    d['return_3d'] = d['MovementPercent'].rolling(3).sum()
    d['volatility_5d'] = d['MovementPercent'].rolling(5).std()
    return d

def calculate_metrics(name, y_true, y_pred, dates):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    return {
        'model': name,
        'test_start': str(pd.Timestamp(dates.min()).date()),
        'test_end': str(pd.Timestamp(dates.max()).date()),
        'n_test': int(len(y_true)),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'tn': int(cm[0,0]), 'fp': int(cm[0,1]), 'fn': int(cm[1,0]), 'tp': int(cm[1,1])
    }

def run_li_lstm(df):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    set_seed(42)
    torch.set_num_threads(2)
    print('\nPreparing technical + sentiment features...')
    d = add_technical_features(df)
    features = [
        'MovementPercent','Open','High','Low','Close','Volume',
        'positive_score','negative_score','neutral_score','tweet_count',
        'movement_lag1','movement_mean_3','movement_mean_7','movement_std_7',
        'volume_change','return_3d','volatility_5d'
    ]
    print(f'Number of input features: {len(features)}')
    d = d.dropna(subset=features + ['Target1d']).reset_index(drop=True)
    print(f'Usable rows after feature preparation: {len(d)}')
    train_rows = d['Date'] <= TRAIN_END
    scaler = StandardScaler()
    scaler.fit(d.loc[train_rows, features])
    values = scaler.transform(d[features]).astype(np.float32)
    targets = d['Target1d'].astype(np.float32).to_numpy()
    dates = d['Date'].to_numpy()
    LOOKBACK = 10
    X, y, sample_dates = [], [], []
    for i in range(LOOKBACK-1, len(d)):
        X.append(values[i-LOOKBACK+1:i+1]); y.append(targets[i]); sample_dates.append(dates[i])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    sequence_dates = pd.to_datetime(sample_dates)
    print(f'Sequence length: {LOOKBACK} days')
    print(f'Total sequences: {len(X)}')
    train_mask = sequence_dates <= TRAIN_END
    validation_mask = (sequence_dates > TRAIN_END) & (sequence_dates <= VAL_END)
    test_mask = sequence_dates >= TEST_START
    print(f'\nTraining sequences:   {train_mask.sum()}')
    print(f'Validation sequences: {validation_mask.sum()}')
    print(f'Test sequences:       {test_mask.sum()}')

    class TwoLayerLSTM(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.lstm = nn.LSTM(input_size=n_features, hidden_size=32, num_layers=2, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(32, 1)
        def forward(self, x):
            output, _ = self.lstm(x)
            return self.fc(output[:, -1, :]).squeeze(1)

    model = TwoLayerLSTM(X.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCEWithLogitsLoss()
    X_train, y_train = torch.tensor(X[train_mask]), torch.tensor(y[train_mask])
    X_val, y_val = torch.tensor(X[validation_mask]), torch.tensor(y[validation_mask])
    X_test, y_test = torch.tensor(X[test_mask]), torch.tensor(y[test_mask])
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)
    best_state = None; best_val = float('inf'); bad = 0; patience = 8
    print('\nTraining two-layer LSTM...')
    for epoch in range(40):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(); loss = loss_function(model(xb), yb); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): val_loss = loss_function(model(X_val), y_val).item()
        print(f'Epoch {epoch+1:02d} | Validation loss: {val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print('Early stopping.'); break
    if best_state is not None: model.load_state_dict(best_state)
    print('\nPredicting test set...')
    model.eval()
    with torch.no_grad(): probs = torch.sigmoid(model(X_test)).cpu().numpy()
    pred = (probs >= 0.5).astype(int)
    return calculate_metrics('Li et al. (2020) - Two-layer LSTM Technical+Sentiment', y_test.cpu().numpy().astype(int), pred, sequence_dates[test_mask].to_numpy())

def main():
    print('\n' + '='*65)
    print('LI ET AL. (2020) - TWO-LAYER LSTM TECHNICAL + SENTIMENT')
    print('='*65)
    if not DATA.exists(): raise FileNotFoundError(f'\nDataset not found:\n{DATA}')
    print(f'\nLoading dataset:\n{DATA}')
    df = pd.read_csv(DATA)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    print(f'Dataset rows: {len(df)}')
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    required = {'Date','MovementPercent','Open','High','Low','Close','Volume','positive_score','negative_score','neutral_score','tweet_count','Target1d'}
    missing = required - set(df.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    result = run_li_lstm(df)
    out = pd.DataFrame([result])
    out.to_csv(OUTPUT, index=False)
    print('\n' + '='*65); print('RESULT'); print('='*65); print()
    print(out.to_string(index=False, float_format=lambda x:f'{x:.4f}'))
    print(f'\nResults saved to:\n{OUTPUT}')
    print('='*65)

if __name__ == '__main__': main()
