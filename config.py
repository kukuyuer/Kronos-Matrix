# -*- coding: utf-8 -*-
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据仓库路径
DATA_REPO = os.path.join(BASE_DIR, 'data_repo')
DIR_DAILY = os.path.join(DATA_REPO, 'daily')
DIR_MIN5 = os.path.join(DATA_REPO, 'min5')
DIR_MIN15 = os.path.join(DATA_REPO, 'min15')
DIR_MIN30 = os.path.join(DATA_REPO, 'min30')
DIR_MIN60 = os.path.join(DATA_REPO, 'min60')
DIR_WEEKLY = os.path.join(DATA_REPO, 'weekly')
DIR_MARKET = os.path.join(DATA_REPO, 'market_info')

# K线类型映射 (Name: 显示名称, ef_code: efinance代码)
K_TYPE_MAP = {
    '101': {'name': '日 K 线',    'path': DIR_DAILY, 'ef_code': 101, 'ak_freq': 'daily'},
    '102': {'name': '周 K 线', 'path': DIR_WEEKLY, 'ef_code': 102, 'ak_freq': 'weekly'},
    '5':   {'name': '5 分钟线',   'path': DIR_MIN5,  'ef_code': 5,   'ak_freq': '5'},
    '15':  {'name': '15 分钟线',  'path': DIR_MIN15, 'ef_code': 15,  'ak_freq': '15'},
    '30':  {'name': '30 分钟线',  'path': DIR_MIN30, 'ef_code': 30,  'ak_freq': '30'},
    '60':  {'name': '60 分钟线',  'path': DIR_MIN60, 'ef_code': 60,  'ak_freq': '60'}
}

LLM_CONFIG = {
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "api_key": "" 
}

# Application-specific default configurations
APP_CONFIG = {
    "DEFAULT_STOCK_CODE": "600519",
    "LLM_PROVIDERS": ["Google Gemini (官方SDK)", "Google (OpenAI协议)", "DeepSeek", "OpenAI", "自定义"],
    "DEFAULT_MODEL_NAME": "gemini-1.5-flash",
    "DEFAULT_FONT_FILE": "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"
}