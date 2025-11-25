# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pandas_ta as ta

class FeatureEngineer:
    @staticmethod
    def add_technical_indicators(df):
        df = df.copy()
        # 基础指标
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_60'] = ta.sma(df['close'], length=60)
        df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
        df['RSI'] = ta.rsi(df['close'], length=14)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['Upper_20'] = df['high'].rolling(20).max()
        df['Lower_20'] = df['low'].rolling(20).min()
        
        # --- 新增：MACD ---
        # fast=12, slow=26, signal=9
        macd = ta.macd(df['close'])
        df['MACD_line'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['MACD_hist'] = macd['MACDh_12_26_9']
        
        # --- 新增：KDJ ---
        # 9, 3, 3
        kdj = ta.kdj(df['high'], df['low'], df['close'])
        df['K'] = kdj['K_9_3']
        df['D'] = kdj['D_9_3']
        df['J'] = kdj['J_9_3']
        
        return df

class MarketRegimeDetector:
    @staticmethod
    def detect(df):
        current = df.iloc[-1]
        adx = current['ADX']
        trend_status = "趋势市" if adx > 25 else "震荡市"
        atr_rank = df['ATR'].rank(pct=True).iloc[-1]
        vol_status = "高波动" if atr_rank > 0.7 else ("低波动" if atr_rank < 0.3 else "正常波动")
        return trend_status, vol_status

class AlphaModels:
    
    @staticmethod
    def trend_model(df):
        curr = df.iloc[-1]
        if curr['close'] > curr['SMA_20'] > curr['SMA_60']: return 1.0
        elif curr['close'] < curr['SMA_20'] < curr['SMA_60']: return -1.0
        return 0.0

    @staticmethod
    def mean_reversion_model(df):
        curr = df.iloc[-1]
        if curr['RSI'] < 30: return 1.0 
        if curr['RSI'] > 70: return -1.0
        return 0.0

    @staticmethod
    def momentum_model(df):
        """
        新增：动量模型 (MACD + KDJ)
        """
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        score = 0.0
        
        # MACD 金叉/死叉
        # 金叉: 柱子由负转正 或 快线上穿慢线
        if curr['MACD_hist'] > 0 and prev['MACD_hist'] <= 0:
            score += 0.5
        elif curr['MACD_hist'] < 0 and prev['MACD_hist'] >= 0:
            score -= 0.5
        elif curr['MACD_hist'] > 0: # 持续多头
            score += 0.2
        elif curr['MACD_hist'] < 0: # 持续空头
            score -= 0.2
            
        # KDJ 金叉/死叉
        if curr['K'] > curr['D'] and prev['K'] <= prev['D']:
            score += 0.5
        elif curr['K'] < curr['D'] and prev['K'] >= prev['D']:
            score -= 0.5
            
        # J值超买超卖
        if curr['J'] < 0: score += 0.3 # 超卖
        if curr['J'] > 100: score -= 0.3 # 超买
        
        # 归一化到 -1 ~ 1
        return max(min(score, 1.0), -1.0)

    @staticmethod
    def kronos_ai_model(rise_prob):
        if rise_prob > 0.6: return 1.0
        if rise_prob > 0.55: return 0.5
        if rise_prob < 0.4: return -1.0
        if rise_prob < 0.45: return -0.5
        return 0.0

class RiskManager:
    @staticmethod
    def calculate_position(final_score, vol_status):
        base_pos = abs(final_score)
        if vol_status == "高波动": base_pos *= 0.5
        elif vol_status == "低波动": base_pos *= 1.2
        base_pos = min(base_pos, 1.0)
        
        action = "观望"
        if final_score > 0.2: action = "买入/持有"
        elif final_score < -0.2: action = "卖出/空仓"
        
        return action, base_pos

class StrategyEngine:
    def __init__(self, df):
        self.df = FeatureEngineer.add_technical_indicators(df)
        
    def run_analysis(self, kronos_rise_prob=0.5):
        trend_status, vol_status = MarketRegimeDetector.detect(self.df)
        
        s_trend = AlphaModels.trend_model(self.df)
        s_mean = AlphaModels.mean_reversion_model(self.df)
        s_mom = AlphaModels.momentum_model(self.df) # 新增动量信号
        s_kronos = AlphaModels.kronos_ai_model(kronos_rise_prob)
        
        # 权重分配 (加入动量)
        # Kronos: 40%, 趋势: 20%, 回归: 20%, 动量: 20%
        w_kronos = 0.4
        w_trend = 0.2
        w_mean = 0.2
        w_mom = 0.2
        
        final_score = (s_trend * w_trend) + (s_mean * w_mean) + (s_mom * w_mom) + (s_kronos * w_kronos)
        
        action, pos_pct = RiskManager.calculate_position(final_score, vol_status)
        
        curr = self.df.iloc[-1]
        
        return {
            "regime": {"trend": trend_status, "volatility": vol_status},
            "signals": {"Trend": s_trend, "RSI": s_mean, "Momentum(KDJ/MACD)": s_mom, "Kronos": s_kronos},
            "weights": {"w_kronos": w_kronos, "w_trend": w_trend, "w_mean": w_mean, "w_mom": w_mom},
            "final_score": final_score,
            "decision": {"action": action, "position_pct": pos_pct},
            "key_levels": {"support": curr['Lower_20'], "resistance": curr['Upper_20']},
            "processed_df": self.df
        }