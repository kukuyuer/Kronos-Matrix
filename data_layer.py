

# -*- coding: utf-8 -*-
import pandas as pd
import os
import config
import requests
import json

class DataLayer:
    """数据层：东方财富 K 线接口 (替代新浪)"""
    
    def __init__(self):
        self.ADJUST_MAP = {"前复权": 1, "后复权": 2, "不复权": 0}
        self.SUFFIX_MAP = {"前复权": "qfq", "后复权": "hfq", "不复权": "none"}

    def _get_eastmoney_kline(self, code, k_type, adjust_type):
        """
        调用东方财富 K 线接口 (替代原新浪)
        """
        # 1. 市场标识转换 (1=沪, 0=深/北)
        # 00开头(深), 30开头(创业), 60/68开头(沪), 4/8开头(北)
        secid_prefix = "1" if str(code).startswith("6") else "0"
        secid = f"{secid_prefix}.{code}"

        # 2. 周期转换 
        # Config映射: 101=日, 102=周, 5=5分...
        # 东财映射: 101=日, 102=周, 103=月, 5=5分, 15=15分, 30=30分, 60=60分
        klt = str(k_type) if str(k_type) in ['5', '15', '30', '60', '101', '102'] else '101'

        # 3. 复权转换 (1=前复权, 2=后复权, 0=不复权)
        fqt_map = {"前复权": "1", "后复权": "2", "不复权": "0"}
        fqt = fqt_map.get(adjust_type, "1")

        # 4. 构建URL
        # f51:日期, f52:开, f53:收, f54:高, f55:低, f56:量, f57:额
        fields = "f51,f52,f53,f54,f55,f56,f57"
        # lmt=1023 获取最近1023根
        url = f"http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&klt={klt}&fqt={fqt}&lmt=1023&end=20500101&iscca=1&fields1=f1,f2,f3,f4,f5,f6,f7,f8&fields2={fields}"

        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data_json = resp.json()
                if data_json and data_json.get('data') and data_json['data'].get('klines'):
                    rows = data_json['data']['klines']
                    parsed_data = []
                    for row in rows:
                        vals = row.split(',')
                        if len(vals) >= 7:
                            parsed_data.append({
                                'timestamps': vals[0],
                                'open': float(vals[1]),
                                'close': float(vals[2]),
                                'high': float(vals[3]),
                                'low': float(vals[4]),
                                'volume': float(vals[5]),
                                'amount': float(vals[6])
                            })
                    
                    df = pd.DataFrame(parsed_data)
                    df['timestamps'] = pd.to_datetime(df['timestamps'])
                    return df
        except Exception as e:
            print(f"EastMoney K-Line Error: {e}")
        
        return pd.DataFrame()

    def get_kline(self, stock_code, k_type='101', source='eastmoney', adjust='前复权', force_update=False):
        """
        获取 K 线 (强制走东方财富通道)
        """
        conf = config.K_TYPE_MAP.get(str(k_type))
        if not conf: return None, {}
        
        suffix = self.SUFFIX_MAP.get(adjust, "qfq")
        file_name = f"{stock_code}_{suffix}.csv"
        file_path = os.path.join(conf['path'], file_name)
        
        # 1. 检查本地缓存
        if os.path.exists(file_path) and not force_update:
            # 简单策略：如果文件存在且非强制更新，可视为有效（生产环境可加时间判断）
            # 这里为了保证数据新鲜，如果有force_update会跳过
            pass 

        # 2. 强制下载 (使用东方财富)
        # print(f"📥 [EastMoney] 下载 {stock_code} ({k_type})...")
        df = self._get_eastmoney_kline(stock_code, k_type, adjust)
        
        if df is not None and not df.empty:
            # 确保目录存在
            os.makedirs(conf['path'], exist_ok=True)
            df.to_csv(file_path, index=False)
            return file_path, {
                "status": "realtime",
                "last_time": str(df['timestamps'].iloc[-1]),
                "rows": len(df)
            }
        
        # 3. 失败回退本地
        if os.path.exists(file_path):
             return file_path, {"status": "cache_stale", "last_time": "unknown", "rows": 0}
             
        return None, {}