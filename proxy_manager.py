# -*- coding: utf-8 -*-
import random
from config_manager import ConfigManager

class ProxyManager:
    _instance = None
    
    active_proxies = []
    cooldown_proxies = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ProxyManager, cls).__new__(cls)
            cls._instance._load_proxies()
        return cls._instance

    def _load_proxies(self):
        """从配置加载代理"""
        config = ConfigManager.load_config()
        saved_list = config.get("proxy_pool", [])
        # 保持列表顺序，但去重
        seen = set()
        self.active_proxies = [x for x in saved_list if not (x in seen or seen.add(x))]
        self.cooldown_proxies = []
        print(f"🔌 [ProxyManager] 已加载 {len(self.active_proxies)} 个代理")

    def get_proxy(self):
        """
        从活跃列表中随机获取一个代理
        """
        if not self.active_proxies:
            return None
        return random.choice(self.active_proxies)

    def mark_success(self, proxy_url):
        """
        [新增] 标记代理成功：将其移到列表末尾
        """
        if proxy_url in self.active_proxies:
            # 为了线程安全和避免索引错误，先移除再添加
            self.active_proxies.remove(proxy_url)
            self.active_proxies.append(proxy_url)
            # print(f"♻️ [ProxyManager] 代理 {proxy_url} 成功，已移至队尾")
            self._save_to_config()

    def mark_failure(self, proxy_url):
        """标记代理失败，移入冷却池"""
        if proxy_url in self.active_proxies:
            self.active_proxies.remove(proxy_url)
            if proxy_url not in self.cooldown_proxies:
                self.cooldown_proxies.append(proxy_url)
            print(f"❄️ [ProxyManager] 代理冷却: {proxy_url} (剩余活跃: {len(self.active_proxies)})")
            self._save_to_config()

    def reset_cooldown(self):
        """手工将冷却池的代理恢复到活跃池"""
        count = len(self.cooldown_proxies)
        if count > 0:
            self.active_proxies.extend(self.cooldown_proxies)
            self.cooldown_proxies = []
            print(f"🔥 [ProxyManager] 已恢复 {count} 个冷却代理")
            self._save_to_config()
        return count

    def add_proxies(self, proxy_list):
        """添加新代理并保存"""
        added_count = 0
        for p in proxy_list:
            p = p.strip()
            if p and p not in self.active_proxies and p not in self.cooldown_proxies:
                self.active_proxies.append(p)
                added_count += 1
        self._save_to_config()
        return added_count

    def _save_to_config(self):
        # 保存顺序：活跃在前，冷却在后
        all_proxies = self.active_proxies + self.cooldown_proxies
        ConfigManager.save_config(proxy_pool=all_proxies)

    def get_status(self):
        return {
            "active": len(self.active_proxies),
            "cooldown": len(self.cooldown_proxies),
            "list_active": self.active_proxies,
            "list_cooldown": self.cooldown_proxies
        }