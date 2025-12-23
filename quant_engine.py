# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pandas_ta as ta

class FeatureEngineer:
    """特征工程"""
    @staticmethod
    def add_technical_indicators(df):
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        # 基础指标 (SMA)
        # 兼容次新股：如果数据不够长，pandas_ta 会产生 NaN，后续逻辑需处理
        df['SMA_20'] = ta.sma(df['close'], length=20)
        df['SMA_60'] = ta.sma(df['close'], length=60)
        
        # ADX (趋势强度)
        try:
            adx = ta.adx(df['high'], df['low'], df['close'], length=14)
            if adx is not None and 'ADX_14' in adx.columns:
                df['ADX'] = adx['ADX_14']
            else:
                df['ADX'] = 0
        except: 
            df['ADX'] = 0
        
        # RSI
        df['RSI'] = ta.rsi(df['close'], length=14)
        
        # ATR (波动率)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # 唐奇安通道 (20日高低点)
        df['Upper_20'] = df['high'].rolling(20).max()
        df['Lower_20'] = df['low'].rolling(20).min()
        
        # MACD
        macd = ta.macd(df['close'])
        if macd is not None:
            df['MACD_line'] = macd['MACD_12_26_9']
            df['MACD_signal'] = macd['MACDs_12_26_9']
            df['MACD_hist'] = macd['MACDh_12_26_9']
        else:
            df['MACD_line'] = 0
            df['MACD_signal'] = 0
            df['MACD_hist'] = 0
        
        # KDJ
        kdj = ta.kdj(df['high'], df['low'], df['close'])
        if kdj is not None:
            df['K'] = kdj['K_9_3']
            df['D'] = kdj['D_9_3']
            df['J'] = kdj['J_9_3']
        else:
            df['K'] = 50
            df['D'] = 50
            df['J'] = 50
            
        return df

class MarketRegimeDetector:
    """市场状态识别"""
    @staticmethod
    def detect(df):
        if df.empty or len(df) < 20:
            return "数据不足", "数据不足"

        current = df.iloc[-1]
        
        # 1. 趋势状态 (基于ADX)
        adx = current.get('ADX')
        if pd.isna(adx): adx = 0
        trend_status = "趋势市" if adx > 25 else "震荡市"
        
        # 2. 波动状态 (基于ATR分位)
        atr = df['ATR']
        if atr.isna().all():
            vol_status = "正常波动"
        else:
            # 获取最近有效ATR
            valid_atr = atr.dropna()
            if valid_atr.empty:
                vol_status = "正常波动"
            else:
                curr_atr = valid_atr.iloc[-1]
                # 计算百分位 (Rank)
                try:
                    # 使用 pandas 的 rank
                    rank = valid_atr.rank(pct=True).iloc[-1]
                    if rank > 0.7: vol_status = "高波动"
                    elif rank < 0.3: vol_status = "低波动"
                    else: vol_status = "正常波动"
                except:
                    vol_status = "正常波动"
                    
        return trend_status, vol_status

class AlphaModels:
    
    @staticmethod
    def trend_model(df):
        """趋势模型：基于均线排列"""
        if df.empty: return 0.0
        curr = df.iloc[-1]
        
        c = curr.get('close')
        ma20 = curr.get('SMA_20')
        ma60 = curr.get('SMA_60')
        
        # 安全检查：如果数据不足导致均线为空，则返回中性
        if pd.isna(c) or pd.isna(ma20) or pd.isna(ma60):
            return 0.0
            
        if c > ma20 > ma60: return 1.0
        elif c < ma20 < ma60: return -1.0
        return 0.0

    @staticmethod
    def mean_reversion_model(df):
        """均值回归模型：基于RSI"""
        if df.empty: return 0.0
        curr = df.iloc[-1]
        
        rsi = curr.get('RSI')
        if pd.isna(rsi): return 0.0 # 数据不足
        
        if rsi < 30: return 1.0  # 超卖，看涨
        if rsi > 70: return -1.0 # 超买，看跌
        return 0.0

    @staticmethod
    def momentum_model(df):
        """动量模型 (MACD + KDJ)"""
        if df.empty or len(df) < 2: return 0.0
        
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0.0
        
        # 1. MACD 金叉/死叉
        hist = curr.get('MACD_hist')
        prev_hist = prev.get('MACD_hist')
        
        # 安全检查
        if pd.isna(hist) or pd.isna(prev_hist):
            pass
        else:
            if hist > 0 and prev_hist <= 0: score += 0.5    # 金叉
            elif hist < 0 and prev_hist >= 0: score -= 0.5  # 死叉
            elif hist > 0: score += 0.2
            elif hist < 0: score -= 0.2
            
        # 2. KDJ 金叉/死叉
        k = curr.get('K')
        d = curr.get('D')
        prev_k = prev.get('K')
        prev_d = prev.get('D')
        
        if pd.isna(k) or pd.isna(d) or pd.isna(prev_k) or pd.isna(prev_d):
            pass
        else:
            if k > d and prev_k <= prev_d: score += 0.5
            elif k < d and prev_k >= prev_d: score -= 0.5
            
        return max(min(score, 1.0), -1.0)

    @staticmethod
    def kronos_ai_model(rise_prob):
        """AI 预测模型"""
        if rise_prob is None: return 0.0
        
        if rise_prob > 0.6: return 1.0
        if rise_prob > 0.55: return 0.5
        if rise_prob < 0.4: return -1.0
        if rise_prob < 0.45: return -0.5
        return 0.0

class RiskManager:
    @staticmethod
    def calculate_position(final_score, vol_status):
        base_pos = abs(final_score)
        
        # 波动率调节
        if vol_status == "高波动": base_pos *= 0.5
        elif vol_status == "低波动": base_pos *= 1.2
        
        base_pos = min(base_pos, 1.0)
        
        action = "观望"
        if final_score > 0.2: action = "买入/持有"
        elif final_score < -0.2: action = "卖出/空仓"
        
        return action, base_pos

class StrategyEngine:
    """
    总控引擎
    """
    def __init__(self, df):
        self.df = FeatureEngineer.add_technical_indicators(df)
        
    def run_analysis(self, kronos_rise_prob=0.5):
        if self.df is None or self.df.empty:
            return {
                "regime": {"trend": "未知", "volatility": "未知"},
                "signals": {},
                "weights": {},
                "final_score": 0,
                "decision": {"action": "数据为空", "position_pct": 0},
                "key_levels": {"support": 0, "resistance": 0},
                "processed_df": pd.DataFrame()
            }

        trend_status, vol_status = MarketRegimeDetector.detect(self.df)
        
        # 计算子策略 (内部已包含空值检查)
        s_trend = AlphaModels.trend_model(self.df)
        s_mean = AlphaModels.mean_reversion_model(self.df)
        s_mom = AlphaModels.momentum_model(self.df)
        s_kronos = AlphaModels.kronos_ai_model(kronos_rise_prob)
        
        # 权重
        w_kronos = 0.4
        w_trend = 0.2
        w_mean = 0.2
        w_mom = 0.2
        
        final_score = (s_trend * w_trend) + (s_mean * w_mean) + (s_mom * w_mom) + (s_kronos * w_kronos)
        
        action, pos_pct = RiskManager.calculate_position(final_score, vol_status)
        
        curr = self.df.iloc[-1]
        # 关键位处理
        sup = curr.get('Lower_20', 0)
        res = curr.get('Upper_20', 0)
        if pd.isna(sup): sup = curr.get('low', 0)
        if pd.isna(res): res = curr.get('high', 0)
        
        return {
            "regime": {"trend": trend_status, "volatility": vol_status},
            "signals": {
                "Trend": s_trend, 
                "RSI": s_mean, 
                "Momentum(KDJ/MACD)": s_mom, 
                "Kronos": s_kronos
            },
            "weights": {"w_kronos": w_kronos, "w_trend": w_trend, "w_mean": w_mean, "w_mom": w_mom},
            "final_score": final_score,
            "decision": {"action": action, "position_pct": pos_pct},
            "key_levels": {"support": sup, "resistance": res},
            "processed_df": self.df
        }