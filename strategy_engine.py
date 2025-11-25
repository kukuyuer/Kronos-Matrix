# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from openai import OpenAI
import google.generativeai as genai 

class Backtester:
    """回测引擎 (保持不变)"""
    @staticmethod
    def run_backtest(df, predictions, initial_capital=100000):
        if df is None or predictions is None: return None
        if isinstance(predictions, list):
            try:
                pred_concat = pd.concat(predictions)
                pred_df = pred_concat.groupby('timestamps', as_index=False)['close'].mean()
            except: return None
        else: pred_df = predictions
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        pred_df['timestamps'] = pd.to_datetime(pred_df['timestamps'])
        merged = pd.merge(df, pred_df, on='timestamps', how='inner', suffixes=('_real', '_pred'))
        if merged.empty: return None
        capital = initial_capital
        position = 0 
        history = []
        for i in range(len(merged) - 1):
            today = merged.iloc[i]
            curr_price = today['close_real']
            next_pred = merged.iloc[i+1]['close_pred']
            predicted_change = (next_pred - curr_price) / curr_price
            action = "HOLD"
            if predicted_change > 0.003 and position == 0: 
                position = capital / curr_price
                capital = 0
                action = "BUY"
            elif predicted_change < -0.003 and position > 0:
                capital = position * curr_price
                position = 0
                action = "SELL"
            total_asset = capital + (position * curr_price)
            history.append({'date': today['timestamps'], 'price': curr_price, 'action': action, 'total_asset': total_asset, 'pred_change': predicted_change})
        res_df = pd.DataFrame(history)
        if res_df.empty: return None
        initial = initial_capital
        final = res_df.iloc[-1]['total_asset']
        returns = (final - initial) / initial * 100
        res_df['real_change'] = res_df['price'].pct_change().shift(-1)
        correct_preds = res_df[res_df['pred_change'] * res_df['real_change'] > 0]
        win_rate = len(correct_preds) / len(res_df) * 100 if len(res_df) > 0 else 0
        metrics = {'total_return': returns, 'win_rate': win_rate, 'final_asset': final, 'trade_count': len(res_df[res_df['action'].isin(['BUY', 'SELL'])])}
        return res_df, metrics

class LLMAdvisor:
    """
    支持 OpenAI 和 Google Gemini 的投资顾问
    V17.3 修复版：正确识别 UI 传来的 Provider 字符串
    """
    def __init__(self, api_key, provider="Google Gemini", model_name="gemini-1.5-flash", base_url=None):
        self.provider = provider
        self.client = None
        self.gemini_model = None
        self.is_google_sdk = False # 标记是否使用官方 SDK

        if not api_key: return

        # --- 【核心修复】字符串匹配逻辑 ---
        # 只要 provider 包含 "官方SDK" 或者 严格等于 "Google Gemini"
        if "官方SDK" in provider or provider == "Google Gemini":
            self.is_google_sdk = True
            try:
                genai.configure(api_key=api_key)
                # 尝试初始化指定模型，如果失败则回退
                try:
                    self.gemini_model = genai.GenerativeModel(model_name)
                except:
                    print(f"模型 {model_name} 初始化失败，尝试回退到 gemini-pro")
                    self.gemini_model = genai.GenerativeModel("gemini-pro")
            except Exception as e:
                print(f"Google AI 配置失败: {e}")
        else:
            # OpenAI / DeepSeek / Google(OpenAI协议)
            self.is_google_sdk = False
            default_url = "https://api.deepseek.com"
            if provider == "OpenAI": default_url = "https://api.openai.com/v1"
            
            target_url = base_url if base_url else default_url
            self.client = OpenAI(api_key=api_key, base_url=target_url)
            self.openai_model = model_name if model_name else ("deepseek-chat" if provider=="DeepSeek" else "gpt-3.5-turbo")

    def get_advice(self, stock_code, stock_name, k_data, pred_stats, financial_info, news_text, k_type_name):
        last_close = k_data.iloc[-1]['close']
        
        prompt = f"""
        你是一位顶级基金经理。请对 {stock_name} ({stock_code}) 进行分析。

        【1. 技术与量化】
        - 周期: {k_type_name}
        - 最新价: {last_close:.2f}
        - Kronos预测上涨概率: {pred_stats.get('rise_probability', 0)*100:.1f}%
        - 波动系数: {pred_stats.get('volatility_amplification', 0):.2f}

        【2. 基本面】
        - PE: {financial_info.get('动态市盈率', 'N/A')}
        - 市值: {financial_info.get('总市值', 'N/A')}

        【3. 资讯】
        {news_text}

        请给出决策报告：
        1. **消息面解读**：简述利好/利空。
        2. **综合研判**：短期趋势预测。
        3. **操作策略**：仓位建议（0-100%）和买卖逻辑。
        """

        try:
            # 使用 flag 判断，而不是字符串匹配，更稳健
            if self.is_google_sdk:
                if self.gemini_model:
                    try:
                        response = self.gemini_model.generate_content(prompt)
                        return response.text
                    except Exception as e:
                        return f"⚠️ Google 模型调用错误: {str(e)}"
                return "❌ Google 模型未就绪"
            
            elif self.client:
                response = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )
                return response.choices[0].message.content
            return "❌ Client 未初始化"
        except Exception as e:
            return f"AI 分析中断: {e}"