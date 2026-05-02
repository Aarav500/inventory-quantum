"""Deep forecasting model using Temporal Fusion Transformer concepts."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import timedelta

from app.models.inventory import ForecastPoint


class DeepForecaster:
    """
    Deep learning forecaster using simplified TFT-like architecture.
    
    Uses PyTorch for multi-horizon forecasting with uncertainty quantification
    via quantile regression.
    """
    
    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        quantiles: List[float] = None,
    ):
        """
        Initialize forecaster.
        
        Args:
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            learning_rate: Learning rate
            epochs: Training epochs
            batch_size: Batch size
            quantiles: Quantiles for uncertainty (default: [0.025, 0.5, 0.975])
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.quantiles = quantiles or [0.025, 0.5, 0.975]
        self.last_metrics = None
        self._model = None
        self._scaler = None
    
    def _create_sequences(
        self,
        values: np.ndarray,
        lookback: int = 30
    ) -> tuple:
        """Create input sequences for training."""
        X, y = [], []
        for i in range(lookback, len(values)):
            X.append(values[i-lookback:i])
            y.append(values[i])
        return np.array(X), np.array(y)
    
    def forecast(
        self,
        data: pd.DataFrame,
        horizon: int = 30,
        include_intervals: bool = True,
        confidence: float = 0.95
    ) -> List[ForecastPoint]:
        """
        Generate deep learning forecasts.
        
        Args:
            data: DataFrame with 'date' and 'quantity_sold' columns
            horizon: Forecast horizon in days
            include_intervals: Whether to include prediction intervals
            confidence: Confidence level for intervals
        
        Returns:
            List of ForecastPoint objects
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            from app.forecasting.lightgbm_model import LightGBMForecaster
            lgbm = LightGBMForecaster()
            self.last_metrics = {'error': 'PyTorch not available, falling back to LightGBM'}
            return lgbm.forecast(data, horizon, include_intervals, confidence)
        
        # Prepare time series
        df = data.copy()
        df = df.sort_values('date')
        df['date'] = pd.to_datetime(df['date'])
        ts = df.groupby('date')['quantity_sold'].sum()
        values = ts.values.astype(np.float32)
        
        lookback = min(30, len(values) // 3)
        
        # Check minimum data
        if len(values) < lookback + 20:
            from app.forecasting.lightgbm_model import LightGBMForecaster
            lgbm = LightGBMForecaster()
            self.last_metrics = {'error': 'Insufficient data for deep model'}
            return lgbm.forecast(data, horizon, include_intervals, confidence)
        
        # Normalize
        mean_val = values.mean()
        std_val = values.std() + 1e-8
        normalized = (values - mean_val) / std_val
        
        # Create sequences
        X, y = self._create_sequences(normalized, lookback)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # Add feature dim
        y_tensor = torch.FloatTensor(y)
        
        # Define model
        class QuantileLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout, num_quantiles):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True
                )
                self.fc = nn.Linear(hidden_size, num_quantiles)
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = self.fc(lstm_out[:, -1, :])
                return out
        
        model = QuantileLSTM(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_quantiles=len(self.quantiles)
        )
        
        # Quantile loss
        def quantile_loss(preds, target, quantiles):
            losses = []
            for i, q in enumerate(quantiles):
                errors = target - preds[:, i]
                losses.append(torch.max((q - 1) * errors, q * errors).mean())
            return sum(losses)
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = quantile_loss(preds, batch_y, self.quantiles)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
        self._model = model
        
        # Generate forecasts
        model.eval()
        predictions = []
        last_date = ts.index.max()
        
        # Use last lookback values
        current_seq = normalized[-lookback:].copy()
        
        with torch.no_grad():
            for h in range(1, horizon + 1):
                forecast_date = last_date + timedelta(days=h)
                
                # Prepare input
                x = torch.FloatTensor(current_seq).unsqueeze(0).unsqueeze(-1)
                
                # Predict quantiles
                quantile_preds = model(x).numpy()[0]
                
                # Denormalize
                quantile_preds = quantile_preds * std_val + mean_val
                
                # Get median and intervals
                median_pred = max(0, quantile_preds[1])  # 0.5 quantile
                lower = max(0, quantile_preds[0])  # 0.025 quantile
                upper = quantile_preds[2]  # 0.975 quantile
                
                point = ForecastPoint(
                    date=forecast_date.date() if hasattr(forecast_date, 'date') else forecast_date,
                    predicted=round(median_pred, 2),
                    lower_bound=round(lower, 2) if include_intervals else None,
                    upper_bound=round(upper, 2) if include_intervals else None,
                )
                predictions.append(point)
                
                # Update sequence for next prediction
                new_value = (median_pred - mean_val) / std_val
                current_seq = np.roll(current_seq, -1)
                current_seq[-1] = new_value
        
        self.last_metrics = {
            'model': 'deep_lstm',
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'epochs': self.epochs,
            'quantiles': self.quantiles,
        }
        
        return predictions
