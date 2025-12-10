# -*- coding: utf-8 -*-
import json
import os
from config import LLM_CONFIG # 引入默认LLM配置

CONFIG_FILE = 'user_config_v3.json'

class ConfigManager:
    DEFAULT_CONFIG = {
        "last_provider": "Google Gemini (官方SDK)",
        # 1. 新增持久化股票池
        "watchlist": ["600519", "300750", "000001"], 
        "providers": {
            "Google Gemini (官方SDK)": {
                "api_key": "", "base_url": "", "model": "gemini-1.5-flash",
                "use_proxy": False, "proxy_url": "http://127.0.0.1:7890"
            },
            "DeepSeek": {
                "api_key": LLM_CONFIG.get("api_key", ""), "base_url": LLM_CONFIG.get("base_url", "https://api.deepseek.com"), 
                "model": LLM_CONFIG.get("model", "deepseek-chat"), "use_proxy": False, "proxy_url": ""
            }
        },
        "ai_context": {
            "news": True,
            "tech": True,
            "kronos_frames": ["101"]
        },
        "kronos_params": {
            "lookback": 100,
            "pred_len": 10,
            "n_preds": 5
        }
    }

    @staticmethod
    def _deep_merge(source, destination):
        """递归合并字典，source会覆盖destination中的值"""
        for key, value in source.items():
            if isinstance(value, dict):
                # 获取节点，如果不存在则创建
                node = destination.setdefault(key, {})
                ConfigManager._deep_merge(value, node)
            else:
                destination[key] = value
        return destination

    @staticmethod
    def load_config():
        if not os.path.exists(CONFIG_FILE):
            return ConfigManager.DEFAULT_CONFIG.copy()
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
                # --- 核心修复：使用深度合并逻辑 ---
                # 以默认配置为基础，用用户配置覆盖它
                return ConfigManager._deep_merge(user_data, ConfigManager.DEFAULT_CONFIG.copy())
        except:
            return ConfigManager.DEFAULT_CONFIG.copy()

    @staticmethod
    def save_config(provider_name=None, p_data=None, ai_ctx=None, kronos_params=None, watchlist=None, proxy_pool=None):
        current_data = ConfigManager.load_config()
        
        if provider_name: current_data["last_provider"] = provider_name
        if p_data and provider_name:
            if provider_name not in current_data["providers"]:
                current_data["providers"][provider_name] = {}
            current_data["providers"][provider_name].update(p_data)
        
        if ai_ctx: current_data["ai_context"] = ai_ctx
        if kronos_params: current_data["kronos_params"] = kronos_params
        if watchlist is not None: current_data["watchlist"] = watchlist
        
        # --- 新增: 保存代理池 ---
        if proxy_pool is not None: current_data["proxy_pool"] = proxy_pool
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=4, ensure_ascii=False)