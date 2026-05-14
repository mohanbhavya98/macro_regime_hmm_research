import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import warnings
import os

warnings.filterwarnings("ignore")

class RegimeDetectionModel:
    def __init__(self, n_components=2):
        print(f"🧠 Initializing Hidden Markov Model with {n_components} Regimes...")
        self.n_components = n_components
        self.model = GaussianHMM(n_components=self.n_components, covariance_type="full", n_iter=1000, random_state=42)

    def load_data(self, filename="eurozone_macro_data.csv"):
        print("📂 Loading macroeconomic dataset into memory...")
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
        self.df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return self.df

    def train_model(self):
        print("⚙️ Engineering macro features and training the HMM...")
        
        # FEATURE ENGINEERING: Force the AI to look at macro trends, not daily noise
        # 1. Rolling 14-day Volatility (How chaotic is the market right now?)
        self.df['Stoxx_Vol_14d'] = self.df['Stoxx_Return'].rolling(window=14).std()
        
        # 2. Rolling 14-day Moving Average of VIX changes (Is fear trending up or down?)
        self.df['VIX_Trend_14d'] = self.df['VIX_Change'].rolling(window=14).mean()
        
        # Drop the first 14 days which are now blank (NaN) due to the rolling math
        self.df = self.df.dropna()
        
        # Feed these new, smoothed macro features into the AI
        self.features = self.df[['Stoxx_Return', 'Stoxx_Vol_14d', 'VIX_Trend_14d']].values
        
        self.model.fit(self.features)
        print("✅ Mathematical convergence achieved. AI has learned the Macro Regimes.")

    def predict_regimes(self):
        print("🔍 Predicting the hidden regime for every single trading day...")
        hidden_states = self.model.predict(self.features)
        self.df['Regime'] = hidden_states
        
        # Optional: A neat quant trick to ensure Regime 0 is ALWAYS the high-volatility "Crash" regime
        # If the average volatility of Regime 0 is lower than Regime 1, flip the labels
        vol_regime_0 = self.df[self.df['Regime'] == 0]['Stoxx_Vol_14d'].mean()
        vol_regime_1 = self.df[self.df['Regime'] == 1]['Stoxx_Vol_14d'].mean()
        
        if vol_regime_0 < vol_regime_1:
            self.df['Regime'] = 1 - self.df['Regime']
            
        return self.df
    def visualize_regimes(self):
        print("📊 Generating Regime Visualization Chart...")
        fig, ax = plt.subplots(figsize=(15, 8))
        
        cumulative_stoxx = (1 + self.df['Stoxx_Return']).cumprod()
        ax.plot(cumulative_stoxx.index, cumulative_stoxx, color='black', linewidth=1.5, label='Euro Stoxx 50 (Cumulative)')
        
        regime_colors = {0: 'red', 1: 'green'} 
        
        for i in range(len(self.df) - 1):
            ax.axvspan(self.df.index[i], self.df.index[i+1], 
                       color=regime_colors[self.df['Regime'].iloc[i]], alpha=0.3)
            
        plt.title("Global Market Regimes Identified by Unsupervised Hidden Markov Model", fontsize=16)
        plt.ylabel("Cumulative Returns", fontsize=12)
        plt.legend()
        
        plot_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'regime_plot.png')
        plt.savefig(plot_path)
        print(f"✅ Visualization successfully saved to {plot_path}")

if __name__ == "__main__":
    hmm = RegimeDetectionModel(n_components=2)
    df = hmm.load_data()
    hmm.train_model()
    results_df = hmm.predict_regimes()
    hmm.visualize_regimes()
    
    print("\n--- Final Model Output ---")
    print("Days classified as Regime 0:", len(results_df[results_df['Regime'] == 0]))
    print("Days classified as Regime 1:", len(results_df[results_df['Regime'] == 1]))