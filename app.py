import os
import warnings
import time
import json

# 屏蔽非关键警告
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
warnings.filterwarnings("ignore", message=".*declare_namespace.*")
warnings.filterwarnings("ignore", message=".*use_container_width.*")

import streamlit as st
import pandas as pd
import datetime
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
from openai import OpenAI

# 设置 matplotlib 后端防止 GUI 报错
matplotlib.use('Agg')

# 引入本地模块
from data_layer import DataLayer
from stock_predictor import StockPredictor
from stock_data_provider import StockDataProvider
from config_manager import ConfigManager
from strategy_engine import Backtester, LLMAdvisor
from quant_engine import StrategyEngine
# 引入代理管理器
from proxy_manager import ProxyManager
# 引入市场更新器 
from market_updater import MarketUpdater
import config

# 初始化数据层
repo = DataLayer()

# 页面配置
st.set_page_config(page_title="Kronos  Pro", layout="wide", page_icon="🛡️")
st.title("🛡️ Kronos 终端")

# 加载配置
user_config = ConfigManager.load_config()

# ================== Session State 初始化 ==================
if 'ana_target_code' not in st.session_state: st.session_state.ana_target_code = "600519"
if 'step3_strategy' not in st.session_state: st.session_state.step3_strategy = None
if 'model_list' not in st.session_state: st.session_state.model_list = []
if 'current_k_path' not in st.session_state: st.session_state.current_k_path = None
if 'data_meta' not in st.session_state: st.session_state.data_meta = {}
if 'selected_model_name' not in st.session_state: st.session_state.selected_model_name = "gemini-1.5-flash"
if 'st_strategy_mode' not in st.session_state: st.session_state.st_strategy_mode = "自定义筛选"

# 筛选参数默认值
defaults = {
    'f_pe_min': 0, 'f_pe_max': 200, 
    'f_cap_min': 0, 
    'f_chg_min': -20.0,
    'f_vr_min': 0.0, 'f_turnover_min': 0.0, 'f_industry': "全部",
    'f_roe_min': 0.0, 'f_div_min': 0.0, 'f_pb_max': 20.0,
    'f_margin_min': 0.0, 'f_g_min': -100.0, 'f_rps_min': 0, 'f_peg_max': 10.0
}
for k, v in defaults.items():
    if k not in st.session_state: st.session_state[k] = v

K_MAP = {"5": "5分钟", "15": "15分钟", "30": "30分钟", "60": "60分钟", "101": "日线"}

# ================== ⚡ 性能优化：缓存装饰器 (支持代理参数) ==================

# 1. 缓存快照数据 (TTL=3600秒)
@st.cache_data(ttl=3600, show_spinner="加载本地数据...")
def get_cached_snapshot(timestamp_key):
    return StockDataProvider.get_market_snapshot_local()

# 2. 缓存实时行情 (TTL=10秒)
@st.cache_data(ttl=10, show_spinner=False)
def get_cached_realtime_info(code, use_proxy):
    return StockDataProvider.get_realtime_info([code], use_proxy=use_proxy)

# 3. 缓存 F10 和静态数据 (TTL=1小时)
@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_f10_data(code, use_proxy):
    f10 = StockDataProvider.get_financial_indicators(code, use_proxy=use_proxy)
    fund = StockDataProvider.get_fundamentals(code, use_proxy=use_proxy)
    esg = StockDataProvider.get_esg_rating(code, use_proxy=use_proxy)
    core = StockDataProvider.get_core_concepts(code, use_proxy=use_proxy)
    return f10, fund, esg, core

# 4. 缓存新闻 (TTL=300秒)
@st.cache_data(ttl=300, show_spinner=False)
def get_cached_news(code, use_proxy):
    return StockDataProvider.get_stock_news(code, top_n=5, use_proxy=use_proxy)

# 5. 缓存资金流 (TTL=600秒)
@st.cache_data(ttl=600, show_spinner=False)
def get_cached_money_flow(code, use_proxy):
    return StockDataProvider.get_money_flow_daily(code, days=20, use_proxy=use_proxy)

# ================== 辅助函数 ==================
def apply_proxy(proxy_url):
    """仅为 LLM 请求设置代理环境变量"""
    if proxy_url:
        os.environ['http_proxy'] = proxy_url; os.environ['https_proxy'] = proxy_url
    else:
        os.environ.pop('http_proxy', None); os.environ.pop('https_proxy', None)

def get_available_models(provider, api_key, base_url=None, proxy=None):
    apply_proxy(proxy)
    models = []
    try:
        if provider == "Google Gemini (官方SDK)":
            genai.configure(api_key=api_key)
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods: models.append(m.name.replace('models/', ''))
        elif provider in ["DeepSeek", "OpenAI", "Google (OpenAI协议)"]:
            url = base_url if base_url else ("https://api.deepseek.com" if provider=="DeepSeek" else "https://api.openai.com/v1")
            client = OpenAI(api_key=api_key, base_url=url)
            models = [m.id for m in client.models.list().data]
    except Exception as e: st.error(f"错误: {e}")
    return sorted(models) if models else []

def run_silent_kronos(code, k_type, data_src, adjust_type, params):
    try:
        time.sleep(0.1)
        csv_path, meta = repo.get_kline(code, k_type=k_type, source=data_src, adjust=adjust_type)
        if csv_path and os.path.exists(csv_path):
            if len(pd.read_csv(csv_path)) < params['lookback']: return None, meta
            pred = StockPredictor(data_file=csv_path, output_dir='./output', plot_file=None, n_predictions=params['n_preds'], lookback=params['lookback'], pred_len=params['pred_len'], stock_code=code, verbose=False)
            res = pred.run_analysis()
            if res and 'statistics' in res: return res['statistics']['close'], meta
    except: pass
    return None, {}

def plot_tech_chart(df, levels, title_suffix=""):
    if df is None or df.empty: return None
    plot_df = df.tail(150).copy()
    plot_df['timestamps'] = pd.to_datetime(plot_df['timestamps'], errors='coerce')
    plot_df = plot_df.dropna(subset=['timestamps'])
    plot_df['date_str'] = plot_df['timestamps'].dt.strftime('%Y-%m-%d %H:%M')
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=plot_df['date_str'], open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name='K线'), row=1, col=1)
    
    if 'Upper_20' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df['date_str'], y=plot_df['Upper_20'], line=dict(color='rgba(255,0,0,0.3)', width=1), name='阻力'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df['date_str'], y=plot_df['Lower_20'], line=dict(color='rgba(0,255,0,0.3)', width=1), name='支撑'), row=1, col=1)
    
    if 'MACD_hist' in plot_df.columns:
        colors = ['red' if val >= 0 else 'green' for val in plot_df['MACD_hist']]
        fig.add_trace(go.Bar(x=plot_df['date_str'], y=plot_df['MACD_hist'], marker_color=colors, name='MACD柱'), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_df['date_str'], y=plot_df['MACD_line'], line=dict(color='orange', width=1), name='DIF'), row=2, col=1)
        fig.add_trace(go.Scatter(x=plot_df['date_str'], y=plot_df['MACD_signal'], line=dict(color='blue', width=1), name='DEA'), row=2, col=1)
    
    if 'RSI' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df['date_str'], y=plot_df['RSI'], line=dict(color='#7e57c2', width=1.5), name='RSI'), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", row=3, col=1, line_color="red", line_width=1)
        fig.add_hline(y=30, line_dash="dot", row=3, col=1, line_color="green", line_width=1)

    fig.update_layout(title=f"技术概览 {title_suffix}", height=750, xaxis_rangeslider_visible=False, xaxis_type='category', xaxis={'type':'category','showgrid':False}, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def plot_money_flow(df_flow):
    if df_flow is None or df_flow.empty: return None
    fig = go.Figure()
    colors = ['red' if x > 0 else 'green' for x in df_flow['主力净流入']]
    fig.add_trace(go.Bar(x=df_flow['日期'], y=df_flow['主力净流入'], marker_color=colors, name='主力净流入'))
    fig.update_layout(title="主力资金流向 (近20日)", yaxis_title="净流入 (元)", height=350, margin=dict(l=10, r=10, t=40, b=10))
    return fig

# ================== 侧边栏 ==================
with st.sidebar:
    st.header("🎛️ 总控台")
    
    with st.expander("1. 标的与数据", expanded=True):
        code_input = st.text_input("分析股票代码", value=st.session_state.ana_target_code)
        if code_input != st.session_state.ana_target_code: st.session_state.ana_target_code = code_input
        
        k_labels = [config.K_TYPE_MAP[k]['name'] for k in config.K_TYPE_MAP.keys()]
        selected_k_idx = st.selectbox("主分析周期", range(len(k_labels)), format_func=lambda x: k_labels[x], index=0)
        selected_k_type = list(config.K_TYPE_MAP.keys())[selected_k_idx]
        selected_k_name = k_labels[selected_k_idx]
        adjust_type = st.selectbox("复权", ["前复权", "不复权", "后复权"], index=0)

    saved_params = user_config.get("kronos_params", {})
    with st.expander("2. Kronos 模型参数", expanded=False):
        lookback = st.number_input("Lookback", 50, 500, saved_params.get("lookback", 100))
        pred_len = st.slider("步长", 5, 60, saved_params.get("pred_len", 10))
        n_preds = st.slider("采样", 1, 50, saved_params.get("n_preds", 10))

    last_provider = user_config.get("last_provider", "Google Gemini (官方SDK)")
    with st.expander("3. AI 配置", expanded=True):
        data_src_label = st.radio("K线数据源", ["东方财富 (EastMoney)", "AkShare (备用)"], horizontal=True)
        data_src = "eastmoney" 
        
        force_sync = st.checkbox("强制云端同步")
        st.divider()
        
        llm_provider = st.selectbox("AI 提供商", list(user_config["providers"].keys()), index=list(user_config["providers"].keys()).index(last_provider) if last_provider in user_config["providers"] else 0)
        p_config = user_config["providers"].get(llm_provider, {})
        
        api_key_input = st.text_input("API Key", value=p_config.get("api_key", ""), type="password")
        base_url_input = st.text_input("Base URL", value=p_config.get("base_url", ""))
        use_proxy = st.checkbox("启用 LLM 代理", value=p_config.get("use_proxy", False))
        proxy_url = st.text_input("代理地址", value=p_config.get("proxy_url", "http://127.0.0.1:7890"))
        
        curr_model = p_config.get("model", "gemini-1.5-flash")
        all_models = list(set([curr_model] + st.session_state.model_list))
        all_models.sort()
        idx = all_models.index(curr_model) if curr_model in all_models else 0
        selected_model = st.selectbox("模型", all_models, index=idx)
        st.session_state.selected_model_name = selected_model 
        
        if st.button("🔍 测试连接"):
            if api_key_input:
                with st.spinner("连接..."):
                    mods = get_available_models(llm_provider, api_key_input, base_url_input, proxy_url if use_proxy else None)
                    if mods: st.session_state.model_list = mods; st.success("成功")
                    else: st.error("失败")

        st.markdown("---")
        st.caption("AI 上下文偏好")
        saved_ctx = user_config.get("ai_context", {})
        ctx_news = st.checkbox("包含 F10 资讯", value=saved_ctx.get("news", True))
        ctx_kronos = st.checkbox("包含 Kronos 预测", value=saved_ctx.get("kronos_main", True))
        ctx_tech = st.checkbox("包含技术指标", value=saved_ctx.get("tech", True))
        saved_frames = saved_ctx.get("kronos_frames", ["101"])
        selected_frames = st.multiselect("多周期矩阵推理", options=list(K_MAP.keys()), format_func=lambda x: K_MAP[x], default=saved_frames)

    # === 4. 代理池管理 ===
    with st.expander("4. 🌐 网络与代理池", expanded=False):
        pm = ProxyManager()
        status = pm.get_status()
        
        c_p1, c_p2 = st.columns(2)
        c_p1.metric("活跃代理", status['active'])
        c_p2.metric("冷却中", status['cooldown'], help="请求失败的代理")
        
        new_proxies = st.text_area("添加代理 (http://ip:port)", height=70)
        if st.button("➕ 添加至代理池"):
            if new_proxies:
                plist = new_proxies.strip().split('\n')
                added = pm.add_proxies(plist)
                st.success(f"成功添加 {added} 个代理")
                st.rerun()
        
        if status['cooldown'] > 0:
            if st.button("♻️ 恢复冷却代理"):
                restored = pm.reset_cooldown()
                st.success(f"已恢复 {restored} 个代理")
                st.rerun()

    # === 5. 数据维护  ===
    with st.expander("5. 💾 数据维护", expanded=False):
        st.caption("全市场数据更新 (每日收盘后运行)")
        c_up1, c_up2 = st.columns(2)
        btn_test = c_up1.button("🧪 测试更新", help="仅抓取少量数据，验证网络")
        btn_full = c_up2.button("🚀 全量更新", help="抓取全市场5000+只股票")
        
        # 状态容器
        prog_bar = st.progress(0, text="就绪")
        log_box = st.empty()

        # 回调函数
        def ui_log(msg):
            log_box.info(f"📜 {msg}")
        
        def ui_progress(percent):
            prog_bar.progress(percent, text=f"进度: {int(percent*100)}%")

        if btn_test or btn_full:
            is_test = True if btn_test else False
            updater = MarketUpdater()
            try:
                # 1. 财务更新
                updater.update_financial_data(test_mode=is_test, status_callback=ui_log)
                # 2. 市场快照更新
                updater.update_market_snapshot(
                    test_mode=is_test, 
                    status_callback=ui_log, 
                    progress_callback=ui_progress
                )
                st.success("✅ 数据更新完成！正在刷新...")
                st.cache_data.clear()
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"更新失败: {e}")

    if st.button("💾 保存配置"):
        p_data = {"api_key": api_key_input, "base_url": base_url_input, "model": selected_model, "use_proxy": use_proxy, "proxy_url": proxy_url}
        kronos_p = {"lookback": lookback, "pred_len": pred_len, "n_preds": n_preds}
        ai_ctx = {"news": ctx_news, "tech": ctx_tech, "kronos_main": ctx_kronos, "kronos_frames": selected_frames}
        ConfigManager.save_config(llm_provider, p_data, ai_ctx, kronos_p)
        st.success("已保存")

# ================== 主界面 ==================
tab_screener, tab_analysis = st.tabs(["🔍 策略选股工厂", "📈 深度融合台"])

with tab_screener:
    st.markdown("### ⚔️ 策略选股工厂 (F10增强版)")
    
    # 选股工厂读取本地快照
    f_path = os.path.join(config.DATA_REPO, 'market_snapshot_full.csv')
    f_mtime = os.path.getmtime(f_path) if os.path.exists(f_path) else 0
    df_local, file_time = get_cached_snapshot(f_mtime)
    
    if df_local.empty:
        st.error("❌ 本地数据为空！请在侧边栏运行 [🚀 全量更新]。")
    else:
        last_update = datetime.datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M')
        st.caption(f"📅 数据更新: {last_update} | 池容量: {len(df_local)}")

        def set_strat(mode):
            st.session_state.st_strategy_mode = mode
            st.session_state.f_pe_max = 200; st.session_state.f_cap_min = 0; st.session_state.f_roe_min = 0
            st.session_state.f_div_min = 0; st.session_state.f_rps_min = 0; st.session_state.f_g_min = -100
            
            if mode == "💰 高息红利":
                st.session_state.f_div_min = 4.0; st.session_state.f_pe_max = 15; st.session_state.f_cap_min = 100
            elif mode == "💎 核心资产":
                st.session_state.f_roe_min = 15.0; st.session_state.f_cap_min = 200
            elif mode == "🦄 业绩暴增":
                st.session_state.f_g_min = 30.0
            elif mode == "🚀 强势龙头":
                st.session_state.f_rps_min = 90; st.session_state.f_cap_min = 50

        st.markdown("#### 🎯 一键策略")
        c_b1, c_b2, c_b3, c_b4, c_b5 = st.columns(5)
        if c_b1.button("💰 高息红利"): set_strat("💰 高息红利")
        if c_b2.button("💎 核心资产"): set_strat("💎 核心资产")
        if c_b3.button("🦄 业绩暴增"): set_strat("🦄 业绩暴增")
        if c_b4.button("🚀 强势龙头"): set_strat("🚀 强势龙头")
        if c_b5.button("🔄 重置"): set_strat("自定义筛选")
        
        st.info(f"当前策略: **{st.session_state.st_strategy_mode}**")
        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        pe_max = c1.slider("PE (动) <", 0, 500, st.session_state.f_pe_max, key="f_pe_max")
        pb_max = c2.slider("PB <", 0.0, 20.0, float(st.session_state.f_pb_max), key="f_pb_max")
        peg_max = c3.slider("PEG <", 0.1, 10.0, float(st.session_state.f_peg_max), key="f_peg_max")
        cap_min = c4.number_input("总市值 > (亿)", 0, value=st.session_state.f_cap_min, key="f_cap_min")

        c5, c6, c7, c8 = st.columns(4)
        roe_min = c5.slider("ROE > %", 0.0, 50.0, float(st.session_state.f_roe_min), key="f_roe_min")
        g_min = c6.slider("净利增长 > %", -100.0, 200.0, float(st.session_state.f_g_min), key="f_g_min")
        margin_min = c7.slider("毛利率 > %", 0.0, 100.0, float(st.session_state.f_margin_min), key="f_margin_min")
        div_min = c8.slider("股息率 > %", 0.0, 10.0, float(st.session_state.f_div_min), key="f_div_min")

        c9, c10, c11, c12 = st.columns(4)
        chg_min = c9.slider("涨幅 > %", -20.0, 20.0, st.session_state.f_chg_min, key="f_chg_min")
        rps_min = c10.slider("RPS强度 >", 0, 99, st.session_state.f_rps_min, key="f_rps_min")
        vr_min = c11.number_input("量比 >", 0.0, value=float(st.session_state.f_vr_min), step=0.1, key="f_vr_min")
        turn_min = c12.slider("换手 > %", 0.0, 20.0, float(st.session_state.f_turnover_min), key="f_turnover_min")

        all_inds = ['全部']
        if '所处行业' in df_local.columns:
            raw_inds = df_local['所处行业'].dropna().unique()
            all_inds += sorted([str(x) for x in raw_inds if str(x) != '0'])
        
        try: ind_idx = all_inds.index(st.session_state.f_industry)
        except: ind_idx = 0
        sel_ind = st.selectbox("行业板块", all_inds, index=ind_idx, key="f_industry")

        mask = (df_local['总市值'] >= cap_min * 100000000) & (df_local['动态市盈率'] <= pe_max)
        if rps_min > 0: mask &= (df_local['RPS_60'] >= rps_min)
        if div_min > 0: mask &= (df_local['股息率'] >= div_min)
        if roe_min > 0: mask &= (df_local['ROE'] >= roe_min)
        if g_min > -100: mask &= (df_local['净利增长率'] >= g_min)
        if margin_min > 0: mask &= (df_local['毛利率'] >= margin_min)
        mask &= (df_local['涨跌幅'] >= chg_min)
        if sel_ind != '全部': mask &= (df_local['所处行业'] == sel_ind)

        res_df = df_local[mask].copy()
        st.success(f"筛选结果: {len(res_df)} 只 (总 {len(df_local)})")
        
        disp_cols = ["代码", "股票名称", "最新价", "涨跌幅", "ROE", "股息率", "净利增长率", "毛利率", "RPS_60", "动态市盈率", "总市值", "所处行业"]
        final_disp = [c for c in disp_cols if c in res_df.columns]
        
        event = st.dataframe(
            res_df[final_disp], width="stretch", hide_index=True, selection_mode="single-row", on_select="rerun",
            column_config={
                "代码": st.column_config.TextColumn("代码"), 
                "涨跌幅": st.column_config.NumberColumn("涨幅", format="%.2f %%"),
                "ROE": st.column_config.NumberColumn("ROE", format="%.2f %%"), 
                "股息率": st.column_config.NumberColumn("股息", format="%.2f %%"),
                "净利增长率": st.column_config.NumberColumn("增长", format="%.1f %%"),
                "毛利率": st.column_config.NumberColumn("毛利", format="%.1f %%"),
                "RPS_60": st.column_config.ProgressColumn("RPS强度", min_value=0, max_value=100, format="%.0f"),
                "总市值": st.column_config.ProgressColumn("市值", format="$%d", min_value=0, max_value=1000000000000)
            }
        )
        
        if len(event.selection.rows) > 0:
            row = res_df.iloc[event.selection.rows[0]]
            selected_code = str(row['代码'])
            if st.session_state.ana_target_code != selected_code:
                st.session_state.ana_target_code = selected_code
                st.rerun()
            st.info(f"已锁定: **{row['股票名称']}**。请切换到【深度融合台】。")

with tab_analysis:
    code = st.session_state.ana_target_code
    
    c_info, c_ctrl = st.columns([3, 1])
    with c_ctrl:
        # 新增：代理开关
        enable_analysis_proxy = st.toggle("🌍 启用数据代理", value=False, help="若直连数据缺失，请开启此选项尝试代理池")
    
    # 1. 实时行情 (透传开关状态)
    real_info = get_cached_realtime_info(code, use_proxy=enable_analysis_proxy)
    
    name = real_info.loc[code].get('股票名称', '未知') if not real_info.empty and code in real_info.index else "未知"
    price = real_info.loc[code].get('最新价', 0) if not real_info.empty and code in real_info.index else 0
    pct = real_info.loc[code].get('涨跌幅', 0) if not real_info.empty and code in real_info.index else 0
    
    color_price = "red" if pct >= 0 else "green"
    with c_info:
        st.markdown(f"### 🧬 {name} (`{code}`)  ¥{price}  :<span style='color:{color_price}'>{pct}%</span>  <span style='font-size:0.6em; color:gray'>| {selected_k_name} (东财源)</span>", unsafe_allow_html=True)

    if st.button("🚀 启动主图分析", type="primary", use_container_width=True):
        with st.status("正在分析...", expanded=True) as status:
            st.write(f"1. 同步 {selected_k_name} 数据...")
            csv_path, data_meta = repo.get_kline(code, k_type=selected_k_type, source=data_src, adjust=adjust_type)
            st.session_state.data_meta = data_meta
            
            if csv_path:
                st.write("2. 运行 Kronos 时序预测...")
                df = pd.read_csv(csv_path)
                predictor = StockPredictor(data_file=csv_path, output_dir='./output', plot_file='./output/future.png', n_predictions=n_preds, lookback=lookback, pred_len=pred_len, stock_code=code, verbose=False, save_details=True)
                kronos_res = predictor.run_analysis()
                
                st.write("3. 综合策略引擎计算...")
                rise_prob = kronos_res['statistics']['close'].get('rise_probability', 0.5) if kronos_res else 0.5
                engine = StrategyEngine(df)
                res = engine.run_analysis(kronos_rise_prob=rise_prob)
                
                st.session_state.step3_strategy = res
                st.session_state.step3_strategy['kronos_main'] = kronos_res 
                st.session_state.current_k_path = csv_path
                status.update(label="完成", state="complete", expanded=False)
            else: status.update(label="失败: 数据下载错误", state="error")

    if st.session_state.step3_strategy:
        meta = st.session_state.data_meta
        res = st.session_state.step3_strategy
        kronos_res = st.session_state.step3_strategy['kronos_main']
        rise_prob = kronos_res['statistics']['close'].get('rise_probability', 0.5) if kronos_res else 0.5
        signals = res['signals']
        
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AI看涨概率", f"{rise_prob*100:.1f}%", delta=f"{(rise_prob-0.5)*100:.1f}%")
        c2.metric("综合策略评分", f"{res['final_score']:.2f}")
        mom_val = signals.get('Momentum(KDJ/MACD)', 0)
        c3.metric("技术动量", f"{mom_val:.2f}", delta=f"{mom_val:.2f}")
        act_color = "green" if "买" in res['decision']['action'] else "red"
        c4.markdown(f"#### 📢 :{act_color}[{res['decision']['action']}]")
        
        st.divider()
        st.markdown("#### 📊 财务透视 (F10)")
        
        # 2. 深度财务/ESG (透传开关状态)
        f10_data, fund_info, esg_data, core_info = get_cached_f10_data(code, use_proxy=enable_analysis_proxy)
        
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("ROE (加权)", f"{f10_data.get('roe', '-')}%")
        f2.metric("净利增长", f"{f10_data.get('profit_yoy', '-')}%")
        f3.metric("毛利率", f"{f10_data.get('gross_margin', '-')}%")
        f4.metric("PE (动)", f"{fund_info.get('动态市盈率', '-')}")
        f5.metric("ESG 评级", f"{esg_data.get('rating', '-')}")
        st.caption(f"财报日期: {f10_data.get('report_date', '-')}")

        t1, t2, t3 = st.tabs(["📈 技术全景", "💰 资金博弈", "🔮 Kronos 预测"])
        with t1: st.plotly_chart(plot_tech_chart(res['processed_df'], res['key_levels']), use_container_width=True)
        with t2:
            # 3. 资金流向 (透传开关状态)
            df_flow = get_cached_money_flow(code, use_proxy=enable_analysis_proxy)
            if df_flow is not None and not df_flow.empty: st.plotly_chart(plot_money_flow(df_flow), use_container_width=True)
            else: st.info("暂无资金流向数据")
        with t3: 
            if os.path.exists('./output/future.png'): st.image('./output/future.png')
            with st.expander("Kronos 详细统计"):
                if 'report_text' in kronos_res: st.text(kronos_res['report_text'])

        st.divider()
        st.subheader("📖 深度 F10 档案")
        
        # 4. 新闻 (透传开关状态)
        detailed_news = get_cached_news(code, use_proxy=enable_analysis_proxy)
        
        col_f10_1, col_f10_2 = st.columns([1, 1])
        with col_f10_1:
            st.markdown("**🏷️ 核心题材 & 概念**")
            if core_info.get("concepts"):
                if core_info.get("lead_concept"): st.success(f"🔥 {core_info['lead_concept']}")
                for c in core_info["concepts"][:5]: st.caption(c)
            else: st.caption("暂无题材数据")
            st.markdown("**🏭 主营业务**")
            st.info(core_info.get("business", "暂无"))

        with col_f10_2:
            st.markdown("**📰 F10 价值资讯**")
            st.text_area("最近动态", detailed_news, height=250, disabled=True)

        st.divider()
        st.subheader(f"🤖 {llm_provider} 深度研报")
        
        if api_key_input:
            if st.button("🧠 生成深度研报"):
                apply_proxy(proxy_url if use_proxy else None)
                with st.spinner("AI 正在综合基本面、F10核心题材、资金流与预测数据..."):
                    advisor = LLMAdvisor(api_key_input, provider=llm_provider, model_name=st.session_state.selected_model_name, base_url=base_url_input)
                    
                    ctx_list = []
                    data_time = st.session_state.data_meta.get('last_time', '未知')
                    ctx_list.append(f"【数据时效】{data_time}")
                    ctx_list.append(f"【深度基本面】\nROE: {f10_data.get('roe')}%\n净利增长: {f10_data.get('profit_yoy')}%\n毛利率: {f10_data.get('gross_margin')}%\nPE(动): {fund_info.get('动态市盈率')}\nESG: {esg_data.get('rating')}")
                    
                    biz_ctx = f"主营业务: {core_info.get('business', '无')}"
                    concepts_ctx = "核心概念:\n" + "\n".join(core_info.get('concepts', [])[:3])
                    ctx_list.append(f"【F10 核心题材】\n{biz_ctx}\n{concepts_ctx}")
                    
                    if df_flow is not None and not df_flow.empty:
                        sum_flow = df_flow.tail(5)['主力净流入'].sum() / 10000
                        last_flow = df_flow.iloc[-1]['主力净流入'] / 10000
                        ctx_list.append(f"【资金博弈】\n近5日主力净流入: {sum_flow:.1f} 万元\n最新动向: {'流入' if last_flow>0 else '流出'} {abs(last_flow):.1f}万")
                    
                    if ctx_news: ctx_list.append(f"【F10 价值资讯】\n{detailed_news}")
                    if ctx_tech: ctx_list.append(f"【主技术 ({selected_k_name})】\n动量: {mom_val:.2f}\n状态: {res['regime']['trend']}")
                    
                    k_stats = {}
                    if ctx_kronos and kronos_res:
                        k_stats = kronos_res['statistics']['close']
                        ctx_list.append(f"【AI预测】\n波动放大: {k_stats.get('volatility_amplification', 0):.2f}\n上涨概率: {rise_prob*100:.1f}%")
                    
                    if selected_frames:
                        matrix_ctx = ["【多周期矩阵】"]
                        run_params = {"n_preds": n_preds, "lookback": lookback, "pred_len": pred_len}
                        prog_bar = st.progress(0, text="多周期推理中...")
                        for i, frame in enumerate(selected_frames):
                            frame_name = K_MAP[frame]
                            prog_bar.progress((i + 1) / len(selected_frames), text=f"推理: {frame_name}...")
                            if frame == selected_k_type: continue
                            sub_stats, sub_meta = run_silent_kronos(code, frame, data_src, adjust_type, run_params)
                            if sub_stats:
                                prob = sub_stats.get('rise_probability', 0.5)
                                matrix_ctx.append(f"- {frame_name}: {prob*100:.1f}%")
                        prog_bar.empty()
                        if len(matrix_ctx) > 1: ctx_list.append("\n".join(matrix_ctx))
                    
                    full_ctx = "\n\n".join(ctx_list)
                    df_raw = pd.read_csv(st.session_state.current_k_path)
                    advice = advisor.get_advice(code, name, df_raw, k_stats, fund_info, full_ctx, selected_k_name)
                    st.markdown(advice)
        else:
            st.info("💡 请在左侧配置 API Key 以解锁 AI 机构研报功能。")