# -*- coding: utf-8 -*-
import warnings
import os
import time
import datetime
import random
import math
import requests
import pandas as pd
from tqdm import tqdm 
import config
from proxy_manager import ProxyManager
import concurrent.futures

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import efinance as ef
import akshare as ak

class EastMoneyScraper:
    """东方财富全市场数据抓取 (严禁直连)"""
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "http://quote.eastmoney.com/",
            "Connection": "keep-alive"
        }
        self.pm = ProxyManager()
        self.domains = [
            "4.push2.eastmoney.com",
            "push2.eastmoney.com", 
            "push2his.eastmoney.com" 
        ]

    def _log(self, msg, callback=None):
        """内部日志包装器"""
        if callback:
            callback(msg)
        else:
            print(msg)

    def _http_get_simple(self, url, timeout=5, proxy_url=None):
        # 强制检查：必须有代理
        if not proxy_url:
            return None, "No Proxy Provided (Direct Access Forbidden)"

        proxies = {"http": proxy_url, "https": proxy_url}
        try:
            # 代理请求超时
            resp = requests.get(url, headers=self.headers, timeout=timeout, proxies=proxies)
            if resp.status_code == 200: return resp, None
            return None, f"Status {resp.status_code}"
        except Exception as e: return None, str(e)

    def _fetch_page_worker(self, args):
        page, fs, fields, proxy_url = args
        
        # 安全检查：如果没有代理，直接拒绝执行
        if not proxy_url:
            return None, 0, None

        page_size = 100
        
        # 域名轮询
        shuffled_domains = list(self.domains)
        random.shuffle(shuffled_domains)
        
        success_flag = False
        res_df = None
        res_total = 0
        
        for domain in shuffled_domains:
            url = f"http://{domain}/api/qt/clist/get?pn={page}&pz={page_size}&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&wbp2u=|0|0|0|web&fid=f3&fs={fs}&fields={fields}&_"
            resp, err = self._http_get_simple(url, timeout=5, proxy_url=proxy_url)
            
            if resp:
                try:
                    data = resp.json()
                    if data and 'data' in data and 'diff' in data['data']:
                        raw_list = data['data']['diff']
                        total = data['data'].get('total', 0)
                        if raw_list is not None:
                            res_df = pd.DataFrame(raw_list)
                            res_total = total
                            success_flag = True
                            break
                except: pass
            time.sleep(0.1)

        if success_flag:
            if proxy_url: self.pm.mark_success(proxy_url)
            return res_df, res_total, proxy_url
        else:
            if proxy_url: self.pm.mark_failure(proxy_url)
            return None, 0, proxy_url

    def get_full_market_data_mt(self, test_mode=False, status_callback=None, progress_callback=None):
        """
        全量抓取 (纯代理模式)
        """
        fs = "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048"
        fields = "f12,f14,f2,f3,f9,f20,f100,f8,f10,f6,f23,f24"
        
        self._log(f"🚀 [EastMoney] 启动全量抓取 (严禁直连)...", status_callback)
        
        # 0. 预检：无代理则直接终止
        active_count = len(self.pm.active_proxies)
        if active_count == 0:
            self._log("❌ 错误：代理池为空！已配置为禁止直连，任务终止。", status_callback)
            return pd.DataFrame()

        # 1. 获取元数据 (竞速模式)
        self._log(f"📡 正在获取市场元数据 (活跃代理: {active_count})...", status_callback)
        first_df = None
        total_count = 0
        
        BATCH_SIZE = 30
        while self.pm.active_proxies and first_df is None:
            snapshot = list(self.pm.active_proxies)
            if not snapshot: break
            batch = snapshot[:BATCH_SIZE]
            
            self._log(f"⚡ 代理竞速中... 剩余活跃: {len(self.pm.active_proxies)}", status_callback)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
                futures = {
                    executor.submit(self._fetch_page_worker, (1, fs, fields, proxy)): proxy 
                    for proxy in batch
                }
                for future in concurrent.futures.as_completed(futures):
                    try:
                        res_df, res_total, _ = future.result()
                        if res_df is not None:
                            if first_df is None:
                                first_df = res_df
                                total_count = res_total
                    except: pass
                
                # === 修复点：Pandas DataFrame 不能直接用 if 判断 ===
                if first_df is not None: 
                    break
            
            time.sleep(0.5)

        # 检查是否获取成功
        if first_df is None or first_df.empty:
            self._log("❌ 所有代理均尝试失败，无法连接服务器。已停止。", status_callback)
            return pd.DataFrame()
        
        if test_mode:
            return self._clean_df(first_df)

        page_size = 100
        total_pages = math.ceil(total_count / page_size)
        self._log(f"📊 市场总数: {total_count}，共 {total_pages} 页。启动并发下载...", status_callback)

        # 2. 并发下载
        all_data = [first_df]
        pending_pages = list(range(2, total_pages + 1))
        
        max_rounds = 10
        current_round = 1
        
        if progress_callback: progress_callback(0.0)
        pages_done = 1 

        while pending_pages and current_round <= max_rounds:
            proxy_count = len(self.pm.active_proxies)
            
            # 严格检查：如果没有代理了，直接退出循环，不尝试直连
            if proxy_count == 0:
                self._log("⚠️ 代理池已耗尽，停止后续抓取。", status_callback)
                break

            max_workers = min(proxy_count * 2, 50)
            if max_workers < 1: max_workers = 1
            max_workers = min(max_workers, len(pending_pages))
            
            self._log(f"🔄 Round {current_round}: 补录 {len(pending_pages)} 页 (Threads={max_workers})...", status_callback)

            failed_pages = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for p in pending_pages:
                    # 每次取一个代理，如果取不到(None)，fetch_worker会直接返回失败
                    proxy = self.pm.get_proxy() 
                    ft = executor.submit(self._fetch_page_worker, (p, fs, fields, proxy))
                    futures[ft] = p
                
                for future in concurrent.futures.as_completed(futures):
                    p = futures[future]
                    try:
                        res_df, _, _ = future.result()
                        if res_df is not None:
                            all_data.append(res_df)
                            pages_done += 1
                            if progress_callback:
                                progress = min(pages_done / total_pages, 1.0)
                                progress_callback(progress)
                        else:
                            failed_pages.append(p)
                    except:
                        failed_pages.append(p)
            
            pending_pages = failed_pages
            current_round += 1
            if pending_pages: time.sleep(1)
        
        full_df = pd.concat(all_data, ignore_index=True)
        full_df = full_df.drop_duplicates(subset=['f12'])
        
        completion_rate = len(full_df)/total_count*100
        self._log(f"✅ 抓取结束。实获: {len(full_df)} (覆盖率 {completion_rate:.1f}%)", status_callback)
        
        if progress_callback: progress_callback(1.0)
        
        return self._clean_df(full_df)

    def _clean_df(self, df):
        rename_map = {
            'f12': 'code', 'f14': 'name', 'f2': 'price', 'f3': 'pct_chg',
            'f9': 'pe', 'f20': 'market_cap', 'f100': 'industry',
            'f8': 'turnover', 'f10': 'volume_ratio', 'f6': 'amount',
            'f23': 'pb', 'f24': 'pct_60d'
        }
        df = df.rename(columns=rename_map)
        df = df[df['price'].astype(str) != '-'] 
        for col in ['pe', 'market_cap', 'pb', 'pct_60d', 'price', 'pct_chg', 'turnover', 'amount']:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('-', '0')
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

class MarketUpdater:
    def __init__(self):
        self.repo_dir = config.DATA_REPO
        self.snapshot_file = os.path.join(self.repo_dir, 'market_snapshot_full.csv')
        self.industry_map_file = os.path.join(self.repo_dir, 'industry_map.csv')
        self.financial_file = os.path.join(self.repo_dir, 'financial_map.csv')
        self.scraper = EastMoneyScraper()

    def _log(self, msg, callback=None):
        if callback: callback(msg)
        else: print(msg)

    # 代理执行器 (严格模式：失败不直连)
    def _run_with_proxy(self, func, **kwargs):
        max_retries = 20
        retries = 0
        old_http = os.environ.get('http_proxy')
        old_https = os.environ.get('https_proxy')

        while retries < max_retries:
            proxy_url = self.scraper.pm.get_proxy()
            if not proxy_url: break # 代理池空，直接退出

            os.environ['http_proxy'] = proxy_url
            os.environ['https_proxy'] = proxy_url
            try:
                df = func(**kwargs)
                if df is not None:
                    if proxy_url: self.scraper.pm.mark_success(proxy_url)
                    self._restore_env(old_http, old_https)
                    return df
            except:
                if proxy_url: self.scraper.pm.mark_failure(proxy_url)
            retries += 1
        
        # 循环结束，还原环境，并返回 None (不执行 func 兜底)
        self._restore_env(old_http, old_https)
        return None

    def _restore_env(self, old_http, old_https):
        if old_http: os.environ['http_proxy'] = old_http
        else: os.environ.pop('http_proxy', None)
        if old_https: os.environ['https_proxy'] = old_https
        else: os.environ.pop('https_proxy', None)

    def update_financial_data(self, test_mode=False, status_callback=None):
        self._log("🔄 [任务4] 拉取财务数据 (纯代理模式)...", status_callback)
        if test_mode: return

        try:
            df = self._run_with_proxy(ef.stock.get_all_company_performance)
            if df is not None and not df.empty:
                rename_map = {}
                for col in df.columns:
                    if "代码" in col: rename_map[col] = "code"
                    elif "净利润" in col and "增长" in col: rename_map[col] = "profit_yoy"
                    elif "营业收入" in col and "增长" in col: rename_map[col] = "revenue_yoy"
                    elif "净资产收益率" in col: rename_map[col] = "roe"
                    elif "毛利率" in col: rename_map[col] = "gross_margin"
                df = df.rename(columns=rename_map)
                df = df[['code', 'profit_yoy', 'revenue_yoy', 'roe', 'gross_margin']]
                df['code'] = df['code'].astype(str).str.zfill(6)
                for c in ['profit_yoy', 'revenue_yoy', 'roe', 'gross_margin']:
                    df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
                
                df.to_csv(self.financial_file, index=False)
                self._log(f"✅ 财务数据更新成功。", status_callback)
            else:
                self._log("❌ 财务数据拉取失败 (代理池无效)。", status_callback)
        except Exception as e:
            self._log(f"❌ 异常: {e}", status_callback)

    def update_market_snapshot(self, test_mode=False, status_callback=None, progress_callback=None):
        self._log(f"🔄 [任务1] 更新全市场快照 (纯代理模式)...", status_callback)
        
        df_base = self.scraper.get_full_market_data_mt(
            test_mode=test_mode, 
            status_callback=status_callback, 
            progress_callback=progress_callback
        )
        
        if df_base.empty:
            self._log("❌ 无法获取行情数据，更新中止。", status_callback)
            return

        self._log(f"📋 正在合并数据...", status_callback)
        df_base['code'] = df_base['code'].astype(str).str.zfill(6)

        if not test_mode:
            self._log("🧬 补充股息率 (AkShare+Proxy)...", status_callback)
            try:
                # 同样使用 _run_with_proxy，无代理则跳过
                df_ak = self._run_with_proxy(ak.stock_zh_a_spot_em)
                if df_ak is not None and not df_ak.empty:
                    rename_ak = {}
                    for col in df_ak.columns:
                        if '代码' in col: rename_ak[col] = 'code'
                        elif '股息' in col: rename_ak[col] = 'dividend_yield'
                    df_ak = df_ak.rename(columns=rename_ak)
                    if 'dividend_yield' in df_ak.columns:
                        df_ak = df_ak[['code', 'dividend_yield']]
                        df_ak['code'] = df_ak['code'].astype(str).str.zfill(6)
                        df_base = pd.merge(df_base, df_ak, on='code', how='left')
            except: pass

        if os.path.exists(self.financial_file):
            try:
                df_fin = pd.read_csv(self.financial_file, dtype={'code': str})
                df_base = pd.merge(df_base, df_fin, on='code', how='left')
            except: pass

        target_cols = ['code', 'name', 'price', 'pct_chg', 'pe', 'market_cap', 'industry', 'turnover', 'amount', 'volume_ratio', 'pct_60d', 'profit_yoy', 'revenue_yoy', 'roe', 'gross_margin', 'pb', 'dividend_yield']
        for c in target_cols:
            if c not in df_base.columns:
                if c == 'industry': df_base[c] = '其他'
                elif c in ['name', 'code']: df_base[c] = '-'
                else: df_base[c] = 0

        df_base['industry'] = df_base['industry'].fillna('其他').replace(['-', '0', 0], '其他')
        
        if not test_mode:
            try:
                df_map = df_base[['code', 'industry']].drop_duplicates('code')
                df_map.to_csv(self.industry_map_file, index=False)
            except: pass

        df_base['update_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_base.to_csv(self.snapshot_file, index=False)
        self._log(f"✅ 快照更新成功！已保存至 {self.snapshot_file}", status_callback)

    def update_all_kline_incremental(self, days_back=365): pass

if __name__ == "__main__":
    print("\n🚀 Kronos  数据中心 (纯代理+Bug修复版)")
    choice = input("\n请输入选项 [1.全量 / 2.测试]: ").strip()
    updater = MarketUpdater()
    is_test = (choice == '2')
    updater.update_financial_data(test_mode=is_test)
    updater.update_market_snapshot(test_mode=is_test)