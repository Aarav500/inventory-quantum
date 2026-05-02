"""Configuration management using Pydantic Settings."""

from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # AWS Configuration
    aws_access_key_id: str = Field(default="", description="AWS Access Key ID")
    aws_secret_access_key: str = Field(default="", description="AWS Secret Access Key")
    aws_region: str = Field(default="us-east-1", description="AWS Region")
    s3_bucket: str = Field(default="inventory-data", description="S3 Bucket for data storage")
    
    # Application Settings
    debug: bool = Field(default=False, description="Debug mode")
    api_title: str = Field(default="Inventory Quantum API", description="API title")
    api_version: str = Field(default="1.0.0", description="API version")
    
    # Forecasting Settings
    forecast_horizon: int = Field(default=30, description="Default forecast horizon in days")
    backtest_windows: int = Field(default=5, description="Number of rolling backtest windows")
    
    # QUBO Settings
    qubo_max_quantity: int = Field(default=256, description="Maximum order quantity for QUBO encoding")
    sa_initial_temp: float = Field(default=100.0, description="Simulated annealing initial temperature")
    sa_cooling_rate: float = Field(default=0.995, description="Simulated annealing cooling rate")
    sa_iterations: int = Field(default=10000, description="Simulated annealing iterations")
    
    # RL Settings
    rl_gamma: float = Field(default=0.99, description="RL discount factor")
    rl_cql_alpha: float = Field(default=1.0, description="CQL regularization weight")
    rl_batch_size: int = Field(default=256, description="RL training batch size")
    
    # Monitoring Settings
    drift_psi_threshold: float = Field(default=0.2, description="PSI threshold for drift detection")
    drift_ks_alpha: float = Field(default=0.05, description="KS test significance level")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
