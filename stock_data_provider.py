# -*- coding: utf-8 -*-
import efinance as ef
import akshare as ak
import pandas as pd
import datetime
import os
import config

class StockDataProvider:
    """股票多维数据提供商 (V15.1 修复循环导入版)"""

    @staticmethod
    def _drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Helper to drop duplicate columns."""
        return df.loc[:, ~df.columns.duplicated()]

    @staticmethod
    def get_market_snapshot_local():
        """从本地文件读取全市场快照"""
        snapshot_path = os.path.join(config.DATA_REPO, 'market_snapshot_full.csv')
        
        if os.path.exists(snapshot_path):
            try:
                # 1. 读取本地 CSV (全部按字符串读取，防止丢0)
                df = pd.read_csv(snapshot_path, dtype=str)
                
                # 2. 映射列名
                rename_map = {
                    'code': '代码', 'name': '股票名称', 'price': '最新价',
                    'pct_chg': '涨跌幅', 'pe': '动态市盈率', 'market_cap': '总市值',
                    'industry': '所处行业', 'turnover': '换手率'
                }
                df = df.rename(columns=rename_map)
                
                # --- 去除重复列名 ---
                df = df.loc[:, ~df.columns.duplicated()]
                
                # 3. 代码补全
                if '代码' in df.columns:
                    df['代码'] = df['代码'].astype(str).str.zfill(6)
                
                # 4. 只保留 UI 需要的核心列
                target_cols = ['代码', '股票名称', '最新价', '涨跌幅', '动态市盈率', '总市值', '所处行业', '换手率']
                
                # 补齐可能缺失的列
                for c in target_cols:
                    if c not in df.columns:
                        if c == '所处行业': df[c] = '其他'
                        elif c in ['代码', '股票名称']: df[c] = '-'
                        else: df[c] = '0'
                
                df = df[target_cols]
                
                # 5. 数值类型转换
                numeric_cols = ['最新价', '涨跌幅', '动态市盈率', '总市值', '换手率']
                for col in numeric_cols:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                
                file_time = os.path.getmtime(snapshot_path)
                return df, file_time
                
            except Exception as e:
                print(f"读取本地快照失败: {e}")
                return pd.DataFrame(), 0
        else:
            return pd.DataFrame(), 0

    @staticmethod
    def get_realtime_info(stock_codes):
        """获取实时行情"""
        try:
            df = ef.stock.get_latest_quote(stock_codes)
            if df is None or df.empty: return pd.DataFrame()
            
            # 重命名
            rename_map = {'名称': '股票名称', '代码': '代码', '最新价': '最新价', '涨跌幅': '涨跌幅'}
            df = df.rename(columns=rename_map)
            
            # 去重
            df = df.loc[:, ~df.columns.duplicated()]
            
            # 补0
            if '代码' in df.columns:
                df['代码'] = df['代码'].astype(str).str.zfill(6)
                df = df.set_index('代码')
            
            # 容错
            if '股票名称' not in df.columns:
                df['股票名称'] = df['名称'] if '名称' in df.columns else '未知'
            if '最新价' in df.columns:
                df['最新价'] = pd.to_numeric(df['最新价'], errors='coerce').fillna(0)
            
            return df
        except: return pd.DataFrame()

    @staticmethod
    def get_fundamentals(stock_code):
        """获取基本面"""
        try:
            df = ef.stock.get_latest_quote([stock_code])
            if df is not None and not df.empty:
                rename_map = {'名称': '股票名称'}
                df = df.rename(columns=rename_map)
                df = df.loc[:, ~df.columns.duplicated()] # 去重
                return df.iloc[0].to_dict()
            return {}
        except: return {}

    @staticmethod
    def get_stock_news(stock_code, top_n=10):
        """获取新闻"""
        try:
            news_df = ak.stock_news_em(symbol=stock_code)
            if news_df is not None and not news_df.empty:
                cols = news_df.columns.tolist()
                time_col = next((c for c in cols if '时间' in c), cols[0])
                title_col = next((c for c in cols if '标题' in c), cols[1])
                lines = []
                for _, row in news_df.head(top_n).iterrows():
                    lines.append(f"- [{str(row[time_col])[:16]}] {str(row[title_col])}")
                return "\n".join(lines)
        except: pass
        return "⚠️_NEWS_FAILED"