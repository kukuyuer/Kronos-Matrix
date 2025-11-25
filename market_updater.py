# -*- coding: utf-8 -*-
import warnings
# 屏蔽烦人的第三方库警告
warnings.filterwarnings("ignore", category=UserWarning, module='py_mini_racer')
warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='pkg_resources')

import efinance as ef
import akshare as ak
import pandas as pd
import os
import time
import datetime
from tqdm import tqdm 
import config

class MarketUpdater:
    def __init__(self):
        self.repo_dir = config.DATA_REPO
        self.snapshot_file = os.path.join(self.repo_dir, 'market_snapshot_full.csv')
        self.daily_dir = config.DIR_DAILY
        
        # 配置：每批次处理数量和休息时间，防封IP
        self.BATCH_SIZE = 50 
        self.SLEEP_TIME = 1.5 

    def update_market_snapshot(self):
        """
        任务1：更新全市场快照（用于选股器）
        """
        print("🔄 [任务1] 开始更新全市场快照...")
        try:
            # 1. 尝试 efinance 获取全市场 (速度快)
            df = ef.stock.get_realtime_quotes('沪深A股')
            if df is not None and not df.empty:
                rename_map = {
                    '代码': 'code', '名称': 'name', '最新价': 'price', 
                    '涨跌幅': 'pct_chg', '动态市盈率': 'pe', '总市值': 'market_cap', 
                    '所处行业': 'industry', '成交量': 'volume', '换手率': 'turnover'
                }
                df = df.rename(columns=rename_map)
                
                cols = ['code', 'name', 'price', 'pct_chg', 'pe', 'market_cap', 'industry', 'turnover']
                for c in cols:
                    if c not in df.columns: df[c] = 0
                
                df['update_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                df.to_csv(self.snapshot_file, index=False)
                print(f"✅ 快照更新成功！共 {len(df)} 只股票，已存入 data_repo/market_snapshot_full.csv")
                return df
                
        except Exception as e:
            print(f"⚠️ efinance 接口波动: {e}")
            print("🔄 正在尝试 AkShare 备用接口...")

        # 备用：AkShare
        try:
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                rename_map = {
                    '代码': 'code', '名称': 'name', '最新价': 'price', 
                    '涨跌幅': 'pct_chg', '市盈率-动态': 'pe', '总市值': 'market_cap', 
                    '换手率': 'turnover'
                }
                df = df.rename(columns=rename_map)
                df['industry'] = '其他' # AkShare 此接口不带行业
                df['update_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # 确保列存在
                if 'pe' not in df.columns: df['pe'] = 0
                
                df.to_csv(self.snapshot_file, index=False)
                print(f"✅ (备用源) 快照更新成功！共 {len(df)} 只股票。")
                return df
        except Exception as e:
            print(f"❌ 所有接口均失败: {e}")
            return None

    def update_all_kline_incremental(self, days_back=365):
        """
        任务2：全量/增量更新日K线
        """
        print("\n🔄 [任务2] 开始更新个股K线数据 (增量模式)...")
        
        if os.path.exists(self.snapshot_file):
            df_market = pd.read_csv(self.snapshot_file, dtype={'code': str})
        else:
            print("⚠️ 未找到快照文件，正在先执行任务1...")
            self.update_market_snapshot()
            if os.path.exists(self.snapshot_file):
                df_market = pd.read_csv(self.snapshot_file, dtype={'code': str})
            else:
                print("❌ 无法获取股票列表，任务终止。")
                return

        all_codes = df_market['code'].tolist()
        total = len(all_codes)
        print(f"🎯 目标：更新 {total} 只股票的日线数据")
        print("☕ 这可能需要较长时间，请耐心等待...")
        
        for i in tqdm(range(0, total, self.BATCH_SIZE), desc="进度"):
            batch_codes = all_codes[i : i + self.BATCH_SIZE]
            
            for code in batch_codes:
                self._update_single_stock(code, days_back)
            
            time.sleep(self.SLEEP_TIME)

    def _update_single_stock(self, code, days_back):
        file_path = os.path.join(self.daily_dir, f"{code}_qfq.csv")
        
        try:
            start_date = None
            old_df = pd.DataFrame()
            
            if os.path.exists(file_path):
                try:
                    old_df = pd.read_csv(file_path)
                    if 'timestamps' in old_df.columns and not old_df.empty:
                        last_date = pd.to_datetime(old_df['timestamps']).max()
                        start_date = (last_date + datetime.timedelta(days=1)).strftime("%Y%m%d")
                except: pass
            
            if not start_date:
                start_date = (datetime.datetime.now() - datetime.timedelta(days=days_back)).strftime("%Y%m%d")
            
            end_date = datetime.datetime.now().strftime("%Y%m%d")
            
            if start_date > end_date: return

            df_new = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
            
            if df_new is None or df_new.empty: return

            rename_map = {
                '日期': 'timestamps', '开盘': 'open', '收盘': 'close', 
                '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'amount'
            }
            df_new = df_new.rename(columns=rename_map)
            df_new = df_new[['timestamps', 'open', 'close', 'high', 'low', 'volume', 'amount']]
            
            if not old_df.empty:
                if 'amount' not in old_df.columns: old_df['amount'] = 0
                df_final = pd.concat([old_df, df_new])
                df_final = df_final.drop_duplicates(subset=['timestamps'], keep='last')
                df_final = df_final.sort_values('timestamps')
            else:
                df_final = df_new
                
            df_final.to_csv(file_path, index=False)
            
        except:
            pass

if __name__ == "__main__":
    print("\n" + "="*40)
    print("🚀 Kronos 数据中心后台维护程序")
    print("="*40)
    print("1. 仅更新全市场快照 (选股器用, 速度快)")
    print("2. 全量更新 K 线数据 (分析台用, 速度慢)")
    print("3. 同时执行 1 和 2")
    
    choice = input("\n请输入选项 [1/2/3]: ").strip()
    
    updater = MarketUpdater()
    
    if choice == '1':
        updater.update_market_snapshot()
    elif choice == '2':
        updater.update_all_kline_incremental()
    elif choice == '3':
        updater.update_market_snapshot()
        updater.update_all_kline_incremental()
    else:
        print("无效选项，退出。")