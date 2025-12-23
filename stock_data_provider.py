# -*- coding: utf-8 -*-
import efinance as ef
import akshare as ak
import pandas as pd
import datetime
import os
import requests
import json
import config
from proxy_manager import ProxyManager

class StockDataProvider:
    """股票多维数据提供商 (V60.1 真实数据解析适配版)"""
    
    pm = ProxyManager()

    # ================= 核心：网络请求封装 =================
    @staticmethod
    def _http_get(url, params=None, headers=None, timeout=4, max_retries=3, use_proxy=True):
        if not use_proxy:
            try:
                return requests.get(url, params=params, headers=headers, timeout=timeout+2)
            except:
                return None

        retries = 0
        while retries < max_retries:
            proxy_url = StockDataProvider.pm.get_proxy()
            proxies = {"http": proxy_url, "https": proxy_url} if proxy_url else None
            
            try:
                to = timeout + 3 if proxies else timeout
                resp = requests.get(url, params=params, headers=headers, timeout=to, proxies=proxies)
                
                if resp.status_code == 200:
                    if proxy_url: StockDataProvider.pm.mark_success(proxy_url)
                    return resp
                else:
                    if proxy_url: StockDataProvider.pm.mark_failure(proxy_url)
            except:
                if proxy_url: StockDataProvider.pm.mark_failure(proxy_url)
            
            retries += 1
            if not proxy_url and retries >= 2: break
        return None

    @staticmethod
    def _run_with_proxy(func, use_proxy=True, **kwargs):
        if not use_proxy:
            try: return func(**kwargs)
            except: return None

        max_retries = 3
        retries = 0
        old_http = os.environ.get('http_proxy')
        old_https = os.environ.get('https_proxy')

        while retries < max_retries:
            proxy_url = StockDataProvider.pm.get_proxy()
            if proxy_url:
                os.environ['http_proxy'] = proxy_url
                os.environ['https_proxy'] = proxy_url
            else:
                os.environ.pop('http_proxy', None)
                os.environ.pop('https_proxy', None)

            try:
                df = func(**kwargs)
                if df is not None:
                    if proxy_url: StockDataProvider.pm.mark_success(proxy_url)
                    StockDataProvider._restore_env(old_http, old_https)
                    return df
            except:
                if proxy_url: StockDataProvider.pm.mark_failure(proxy_url)
            
            retries += 1
            if not proxy_url and retries >= 1: break
        
        StockDataProvider._restore_env(old_http, old_https)
        return pd.DataFrame()

    @staticmethod
    def _restore_env(old_http, old_https):
        if old_http: os.environ['http_proxy'] = old_http
        else: os.environ.pop('http_proxy', None)
        if old_https: os.environ['https_proxy'] = old_https
        else: os.environ.pop('https_proxy', None)

    # ================= 代码格式转换工具 =================
    @staticmethod
    def _get_secucode(code):
        """DataCenter API: 001309.SZ"""
        code = str(code).zfill(6)
        if code.startswith('6'): return f"{code}.SH"
        if code.startswith('8') or code.startswith('4'): return f"{code}.BJ"
        return f"{code}.SZ"

    @staticmethod
    def _get_market_symbol(code):
        """PageAjax API: SZ001309"""
        code = str(code).zfill(6)
        if code.startswith('6'): return f"SH{code}"
        if code.startswith('8') or code.startswith('4'): return f"BJ{code}"
        return f"SZ{code}"
    
    @staticmethod
    def _get_push2_secid(code):
        """Push2His API: 0.001309"""
        code = str(code).zfill(6)
        if code.startswith('6'): return f"1.{code}"
        return f"0.{code}"

    # ================= 数据获取逻辑 =================

    @staticmethod
    def get_market_snapshot_local():
        snapshot_path = os.path.join(config.DATA_REPO, 'market_snapshot_full.csv')
        if not os.path.exists(snapshot_path): return pd.DataFrame(), 0
        try:
            df = pd.read_csv(snapshot_path, dtype=str)
            rename_map = {
                'code': '代码', 'symbol': '代码', 'name': '股票名称',
                'price': '最新价', 'pct_chg': '涨跌幅', 'pe': '动态市盈率', 
                'market_cap': '总市值', 'industry': '所处行业',
                'turnover': '换手率', 'volume_ratio': '量比', 'amount': '成交额',
                'profit_yoy': '净利增长率', 'revenue_yoy': '营收增长率',
                'roe': 'ROE', 'gross_margin': '毛利率', 'pb': '市净率', 
                'dividend_yield': '股息率', 'pct_60d': '60日涨跌幅'
            }
            df = df.rename(columns=rename_map)
            if '代码' in df.columns: df['代码'] = df['代码'].astype(str).str.zfill(6)
            else: return pd.DataFrame(), 0
            
            numeric_cols = ['最新价', '涨跌幅', '动态市盈率', '总市值', '换手率', '量比', '成交额',
                            '净利增长率', 'ROE', '股息率', '市净率', '60日涨跌幅', '毛利率']
            for col in numeric_cols:
                if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

            if '60日涨跌幅' in df.columns: df['RPS_60'] = df['60日涨跌幅'].rank(pct=True) * 100
            else: df['RPS_60'] = 0.0

            df['PEG'] = 999.0
            valid_peg = (df['动态市盈率'] > 0) & (df['净利增长率'] > 0)
            df.loc[valid_peg, 'PEG'] = df.loc[valid_peg, '动态市盈率'] / df.loc[valid_peg, '净利增长率']
            
            final_cols = ['代码', '股票名称', '最新价', '涨跌幅', '动态市盈率', '总市值', 
                          '所处行业', '换手率', '量比', '成交额', 
                          '净利增长率', 'ROE', '股息率', 'RPS_60', 'PEG', '市净率', '毛利率']
            for c in final_cols:
                if c not in df.columns: df[c] = 0 if c not in ['代码', '股票名称', '所处行业'] else '-'
            return df[final_cols], os.path.getmtime(snapshot_path)
        except: return pd.DataFrame(), 0

    @staticmethod
    def get_realtime_info(stock_codes, use_proxy=True):
        try:
            df = StockDataProvider._run_with_proxy(ef.stock.get_latest_quote, use_proxy=use_proxy, stock_codes=stock_codes)
            if df is None or df.empty: return pd.DataFrame()
            df = df.rename(columns={'名称': '股票名称', '代码': '代码', '最新价': '最新价', '涨跌幅': '涨跌幅'})
            if '代码' in df.columns:
                df['代码'] = df['代码'].astype(str).str.zfill(6)
                return df.set_index('代码')
            return df
        except: return pd.DataFrame()

    @staticmethod
    def get_fundamentals(stock_code, use_proxy=True):
        try:
            df = StockDataProvider._run_with_proxy(ef.stock.get_latest_quote, use_proxy=use_proxy, stock_codes=[stock_code])
            if df is not None and not df.empty:
                return df.rename(columns={'名称': '股票名称'}).iloc[0].to_dict()
        except: pass
        return {}

    @staticmethod
    def _make_datacenter_request(report_name, filter_str, use_proxy=True):
        """DataCenter API"""
        url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
        params = {
            "reportName": report_name, "columns": "ALL", "filter": filter_str,
            "pageNumber": 1, "pageSize": 1, "sortTypes": "-1", "sortColumns": "REPORT_DATE",
            "source": "WEB", "client": "WEB", "_": int(datetime.datetime.now().timestamp() * 1000)
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://data.eastmoney.com/"
        }
        resp = StockDataProvider._http_get(url, params=params, headers=headers, use_proxy=use_proxy)
        if resp:
            try:
                data = resp.json()
                if data.get('success') and data.get('result') and data['result'].get('data'):
                    return data['result']['data']
            except: pass
        return None

    @staticmethod
    def get_financial_indicators(stock_code, use_proxy=True):
        """获取财务透视 (RPT_F10_FINANCE_MAINFINADATA)"""
        res = {"roe": "-", "net_margin": "-", "gross_margin": "-", "profit_yoy": "-", "revenue_yoy": "-", "report_date": "-"}
        secucode = StockDataProvider._get_secucode(stock_code)
        
        # 字段映射已基于真实数据校准
        data = StockDataProvider._make_datacenter_request("RPT_F10_FINANCE_MAINFINADATA", f'(SECUCODE="{secucode}")', use_proxy=use_proxy)
        
        if data:
            latest = data[0]
            res["report_date"] = str(latest.get('REPORT_DATE', '-'))[:10]
            res["roe"] = latest.get('ROEJQ', latest.get('ROEWEIGHTED', '-'))
            res["profit_yoy"] = latest.get('PARENTNETPROFITTZ', latest.get('GSJLR_TBZZ', '-'))
            res["revenue_yoy"] = latest.get('TOTALOPERATEREVETZ', latest.get('YYSR_TBZZ', '-'))
            res["gross_margin"] = latest.get('XSMLL', '-')
            res["net_margin"] = latest.get('XSJLL', '-')
            
        return res

    @staticmethod
    def get_core_concepts(stock_code, use_proxy=True):
        """
        [修正] 获取核心题材与主营 (PageAjax 真实数据解析)
        """
        info = {"concepts": [], "business": "暂无资料", "lead_concept": "暂无"}
        symbol = StockDataProvider._get_market_symbol(stock_code)
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": f"https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?code={symbol}",
            "Host": "emweb.securities.eastmoney.com"
        }

        # 1. 核心题材 (CoreConception)
        # 真实数据结构: {"hxtc": [{"KEYWORD": "经营范围", ...}], "ssbk": [{"BOARD_NAME": "半导体", ...}]}
        try:
            url = f"https://emweb.securities.eastmoney.com/PC_HSF10/CoreConception/PageAjax?code={symbol}"
            resp = StockDataProvider._http_get(url, headers=headers, use_proxy=use_proxy)
            if resp:
                data = resp.json()
                
                # A. 提取所属板块 (ssbk) 作为核心概念标签
                if 'ssbk' in data and data['ssbk']:
                    concepts = []
                    for item in data['ssbk']:
                        name = item.get('BOARD_NAME')
                        if name:
                            concepts.append(f"【{name}】")
                    # 只取前8个最重要的
                    info["concepts"] = concepts[:8]
                    if concepts: info["lead_concept"] = concepts[0].replace('【','').replace('】','')

                # B. 提取主营业务 (从 hxtc 列表中找 KEYWORD="主营业务")
                if 'hxtc' in data and data['hxtc']:
                    for item in data['hxtc']:
                        if item.get('KEYWORD') == '主营业务':
                            content = item.get('MAINPOINT_CONTENT', '')
                            # 清洗一下，去掉太长的废话
                            info["business"] = content[:150] + "..." if len(content) > 150 else content
                            break
        except: pass

        # 2. 详细主营 (备用 - 如果Core接口没拿到)
        if info["business"] == "暂无资料":
            try:
                url_biz = f"https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax?code={symbol}"
                resp = StockDataProvider._http_get(url_biz, headers=headers, use_proxy=use_proxy)
                if resp:
                    data = resp.json()
                    if 'zygc' in data and data['zygc']:
                        # 尝试从主营构成中拼接产品名
                        latest = data['zygc'] 
                        if isinstance(latest, list):
                            # 兼容列表结构
                            flat_list = latest if (len(latest)>0 and 'ITEM_NAME' in latest[0]) else (latest[0] if len(latest)>0 else [])
                            prods = [p.get('ITEM_NAME','') for p in flat_list if isinstance(p, dict) and p.get('ITEM_NAME')]
                            if prods: info["business"] = "主要产品：" + "，".join(prods[:5])
            except: pass
            
        return info

    @staticmethod
    def get_stock_news(stock_code, top_n=5, use_proxy=True):
        """[软数据] 新闻 (PageAjax)"""
        news_text = ""
        symbol = StockDataProvider._get_market_symbol(stock_code)
        url = f"https://emweb.securities.eastmoney.com/PC_HSF10/CompanyNews/PageAjax?code={symbol}"
        headers = {"User-Agent": "Mozilla/5.0", "Referer": f"https://emweb.securities.eastmoney.com/pc_hsf10/pages/index.html?code={symbol}"}
        
        resp = StockDataProvider._http_get(url, headers=headers, use_proxy=use_proxy)
        if resp:
            try:
                data = resp.json()
                if 'news' in data and data['news']:
                    items = data['news'][:top_n]
                    news_list = []
                    for item in items:
                        title = item.get('title', '')
                        date = str(item.get('showtime', ''))[:10]
                        digest = item.get('digest', '')
                        if title:
                            news_list.append(f"• [{date}] {title}\n  {digest}")
                    if news_list:
                        news_text = "\n".join(news_list)
            except: pass
        return news_text if news_text else "暂无最新资讯"

    @staticmethod
    def get_money_flow_daily(stock_code, days=20, use_proxy=True):
        """资金流向 (Push2His)"""
        try:
            secid = StockDataProvider._get_push2_secid(stock_code)
            url = f"http://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get?lmt={days}&klt=101&secid={secid}&fields1=f1&fields2=f51,f52"
            resp = StockDataProvider._http_get(url, use_proxy=use_proxy)
            if resp:
                data = resp.json()
                if data and 'data' in data and 'klines' in data['data']:
                    rows = data['data']['klines']
                    parsed_data = []
                    for row in rows:
                        vals = row.split(',')
                        if len(vals) >= 2:
                            parsed_data.append({'日期': vals[0], '主力净流入': float(vals[1])})
                    if parsed_data:
                        df = pd.DataFrame(parsed_data)
                        df['日期'] = pd.to_datetime(df['日期'])
                        return df
        except: pass
        return pd.DataFrame()
    
    @staticmethod
    def get_esg_rating(stock_code, use_proxy=True):
        try:
            df = StockDataProvider._run_with_proxy(ak.stock_esg_rate_sh_sz, use_proxy=use_proxy, symbol=stock_code)
            if df is not None and not df.empty:
                latest = df.iloc[0]
                return {"rating": latest.get('评级', 'N/A'), "date": str(latest.get('截止日期', '-'))}
        except: pass
        return {"rating": "N/A", "date": "-"}