# -*- coding: utf-8 -*-
import json
import os

CONFIG_FILE = 'user_config_v3.json' # 再次升级文件名，确保配置结构更新

class ConfigManager:
    DEFAULT_CONFIG = {
        "last_provider": "Google Gemini (官方SDK)",
        "providers": {
            "Google Gemini (官方SDK)": {
                "api_key": "", "base_url": "", "model": "gemini-1.5-flash",
                "use_proxy": False, "proxy_url": "http://127.0.0.1:7890"
            },
            "Google (OpenAI协议)": {
                "api_key": "", "base_url": "", "model": "gemini-1.5-flash",
                "use_proxy": False, "proxy_url": "http://127.0.0.1:7890"
            },
            "DeepSeek": {
                "api_key": "", "base_url": "https://api.deepseek.com", "model": "deepseek-chat",
                "use_proxy": False, "proxy_url": ""
            },
            "OpenAI": {
                "api_key": "", "base_url": "https://api.openai.com/v1", "model": "gpt-3.5-turbo",
                "use_proxy": True, "proxy_url": "http://127.0.0.1:7890"
            },
            "自定义": {
                "api_key": "", "base_url": "", "model": "",
                "use_proxy": False, "proxy_url": ""
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
            "n_preds": 10
        }
    }

    @staticmethod
    def load_config():
        if not os.path.exists(CONFIG_FILE):
            return ConfigManager.DEFAULT_CONFIG
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                merged = ConfigManager.DEFAULT_CONFIG.copy()
                # 深度合并 providers
                if "providers" in data:
                    for p, vals in data["providers"].items():
                        if p in merged["providers"]:
                            merged["providers"][p].update(vals)
                # 合并其他顶层键
                for k in ["last_provider", "ai_context", "kronos_params"]:
                    if k in data: merged[k] = data[k]
                return merged
        except:
            return ConfigManager.DEFAULT_CONFIG

    @staticmethod
    def save_config(provider_name, p_data, ai_ctx, kronos_params):
        # 加载旧数据以保留其他 provider 的配置
        current_data = ConfigManager.load_config()
        
        current_data["last_provider"] = provider_name
        
        # 更新当前 provider
        if provider_name not in current_data["providers"]:
            current_data["providers"][provider_name] = {}
        current_data["providers"][provider_name] = p_data
        
        current_data["ai_context"] = ai_ctx
        current_data["kronos_params"] = kronos_params
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(current_data, f, indent=4)