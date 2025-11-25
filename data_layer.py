# -*- coding: utf-8 -*-
import pandas as pd
import os
import efinance as ef
import akshare as ak
import config
import datetime

class DataLayer:
    """V4 数据层：支持复权隔离 & 数据完整性修复"""
    
    def __init__(self):
        # 复权映射：efinance 1=前复权, 2=后复权, 0=不复权
        self.ADJUST_MAP = {
            "前复权": 1,
            "后复权": 2,
            "不复权": 0
        }
        # 文件后缀映射
        self.SUFFIX_MAP = {
            "前复权": "qfq",
            "后复权": "hfq",
            "不复权": "none"
        }

    def get_kline(self, stock_code, k_type='101', source='efinance', adjust='前复权', force_update=False):
        """
        获取K线数据，确保包含 'amount' (成交额)
        """
        conf = config.K_TYPE_MAP.get(str(k_type))
        if not conf: return None
        
        # 文件名加入复权后缀
        suffix = self.SUFFIX_MAP.get(adjust, "qfq")
        adjust_code = self.ADJUST_MAP.get(adjust, 1)
        
        file_name = f"{stock_code}_{suffix}.csv"
        file_path = os.path.join(conf['path'], file_name)
        
        # 1. 检查本地 (如果 force_update 为 False 且文件存在)
        if os.path.exists(file_path) and not force_update:
            # 预读检查是否包含 amount 列
            try:
                check_df = pd.read_csv(file_path, nrows=1)
                if 'amount' in check_df.columns:
                    return file_path
                else:
                    print(f"⚠️ 本地缓存 {file_name} 缺少 amount 列，触发强制更新...")
            except:
                pass # 读取失败也强制更新

        # 2. 下载 (传入复权参数)
        print(f"📥 下载: {stock_code} ({adjust}) Source: {source}")
        try:
            df = pd.DataFrame()
            if source == 'efinance':
                # efinance 的 fqt 参数控制复权
                df = ef.stock.get_quote_history(
                    stock_codes=stock_code, 
                    klt=conf['ef_code'],
                    fqt=adjust_code 
                )
            elif source == 'akshare':
                # AkShare 的 adjust 参数
                ak_adjust = "qfq" if adjust == "前复权" else ("hfq" if adjust == "后复权" else "")
                if conf['ak_freq'] == 'daily':
                    end_d = datetime.datetime.now().strftime("%Y%m%d")
                    start_d = (datetime.datetime.now() - datetime.timedelta(days=365 * 3)).strftime("%Y%m%d") # Fetch 3 years of data by default
                    df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start_d, end_date=end_d, adjust=ak_adjust)
            
            # 3. 数据清洗
            if df is not None and not df.empty:
                rename_map = {
                    '日期': 'timestamps', 'date': 'timestamps',
                    '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount'
                }
                df = df.rename(columns=rename_map)
                
                # 容错处理：如果数据源没有 'amount'，用 收盘价 * 成交量 估算
                if 'amount' not in df.columns:
                    if 'close' in df.columns and 'volume' in df.columns:
                        df['amount'] = df['close'] * df['volume']
                    else:
                        df['amount'] = 0.0

                # 确保保留所有核心列
                cols_to_keep = ['timestamps', 'open', 'close', 'high', 'low', 'volume', 'amount']
                final_cols = [c for c in cols_to_keep if c in df.columns]
                df = df[final_cols]
                
                # 保存
                df.to_csv(file_path, index=False)
                return file_path
                
        except Exception as e:
            print(f"下载/清洗失败: {e}")
            
        # 如果下载失败，且本地有旧文件，尝试返回旧文件
        return file_path if os.path.exists(file_path) else None

    def get_market_list(self):
        path = os.path.join(config.DIR_MARKET, 'stock_list.csv')
        if os.path.exists(path): return pd.read_csv(path, dtype={'代码': str})
        try:
            df = ef.stock.get_latest_quote()
            df.to_csv(path, index=False)
            return df
        except: return pd.DataFrame()