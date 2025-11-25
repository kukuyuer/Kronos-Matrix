import os
import warnings

# 1. 警告屏蔽
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='pkg_resources')
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

matplotlib.use('Agg')

# 引入各模块
from data_layer import DataLayer
from stock_predictor import StockPredictor
from stock_data_provider import StockDataProvider
from config_manager import ConfigManager
from strategy_engine import Backtester, LLMAdvisor
from quant_engine import StrategyEngine
import config

# 初始化
repo = DataLayer()

st.set_page_config(page_title="Kronos V20.3 Matrix", layout="wide", page_icon="🌌")
st.title("🌌 Kronos V20.3 旗舰量化终端")

user_config = ConfigManager.load_config()

# Session State
if 'ana_target_code' not in st.session_state: st.session_state.ana_target_code = "600519"
if 'step1_data' not in st.session_state: st.session_state.step1_data = None
if 'step2_kronos' not in st.session_state: st.session_state.step2_kronos = None
if 'step3_strategy' not in st.session_state: st.session_state.step3_strategy = None
if 'model_list' not in st.session_state: st.session_state.model_list = []
if 'current_k_path' not in st.session_state: st.session_state.current_k_path = None

# K线映射
K_MAP = {"5": "5分钟", "15": "15分钟", "30": "30分钟", "60": "60分钟", "101": "日线"}

# ================== 辅助函数 ==================
def apply_proxy(proxy_url):
    if proxy_url:
        os.environ['http_proxy'] = proxy_url
        os.environ['https_proxy'] = proxy_url
        os.environ['HTTP_PROXY'] = proxy_url
        os.environ['HTTPS_PROXY'] = proxy_url
    else:
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        os.environ.pop('HTTP_PROXY', None)
        os.environ.pop('HTTPS_PROXY', None)

def get_available_models(provider, api_key, base_url=None, proxy=None):
    apply_proxy(proxy)
    models = []
    try:
        if provider == "Google Gemini (官方SDK)":
            genai.configure(api_key=api_key)
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    models.append(m.name.replace('models/', ''))
        elif provider in ["DeepSeek", "OpenAI", "Google (OpenAI协议)"]:
            default_url = "https://api.deepseek.com"
            if provider == "OpenAI": default_url = "https://api.openai.com/v1"
            url = base_url if base_url else default_url
            client = OpenAI(api_key=api_key, base_url=url)
            resp = client.models.list()
            models = [m.id for m in resp.data]
    except Exception as e:
        st.error(f"模型获取失败: {e}")
    return sorted(models) if models else []

def run_silent_kronos(code, k_type, data_src, adjust_type, params):
    """后台静默运行 Kronos"""
    try:
        csv_path = repo.get_kline(code, k_type=k_type, source=data_src, adjust=adjust_type)
        if csv_path and os.path.exists(csv_path):
            pred = StockPredictor(
                data_file=csv_path, output_dir='./output', plot_file=None, 
                n_predictions=params['n_preds'], 
                lookback=params['lookback'], 
                pred_len=params['pred_len'], 
                stock_code=code, verbose=False
            )
            res = pred.run_analysis()
            if res and 'statistics' in res:
                return res['statistics']['close']
    except Exception as e:
        print(f"Silent run error: {e}")
    return None

def plot_tech_chart(df, levels, title_suffix=""):
    if df is None or df.empty: return None
    plot_df = df.tail(150).copy()
    plot_df['timestamps'] = pd.to_datetime(plot_df['timestamps'], errors='coerce')
    plot_df = plot_df.dropna(subset=['timestamps'])
    plot_df['date_str'] = plot_df['timestamps'].dt.strftime('%Y-%m-%d %H:%M')
    
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.6, 0.2, 0.2])
    fig.add_trace(go.Candlestick(x=plot_df['date_str'], open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], name='K线'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date_str'], y=plot_df['Upper_20'], line=dict(color='rgba(255,0,0,0.3)', width=1), name='阻力'), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date_str'], y=plot_df['Lower_20'], line=dict(color='rgba(0,255,0,0.3)', width=1), name='支撑'), row=1, col=1)
    
    colors = ['red' if val >= 0 else 'green' for val in plot_df['MACD_hist']]
    fig.add_trace(go.Bar(x=plot_df['date_str'], y=plot_df['MACD_hist'], marker_color=colors, name='MACD柱'), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date_str'], y=plot_df['MACD_line'], line=dict(color='orange', width=1), name='DIF'), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date_str'], y=plot_df['MACD_signal'], line=dict(color='blue', width=1), name='DEA'), row=2, col=1)
    
    fig.add_trace(go.Scatter(x=plot_df['date_str'], y=plot_df['RSI'], line=dict(color='#7e57c2', width=1.5), name='RSI'), row=3, col=1)
    
    fig.update_layout(title=f"技术概览 {title_suffix}", height=750, xaxis_rangeslider_visible=False, xaxis_type='category', 
                      xaxis={'type':'category','showgrid':False}, xaxis2={'type':'category','showgrid':False}, xaxis3={'type':'category','showgrid':False}, margin=dict(l=10,r=10,t=40,b=10))
    return fig

# ================== 侧边栏 ==================
with st.sidebar:
    st.header("🎛️ 总控台")
    
    # 1. 标的与数据
    with st.expander("1. 标的与数据 (Data)", expanded=True):
        code_input = st.text_input("分析股票代码", value=st.session_state.ana_target_code)
        if code_input != st.session_state.ana_target_code: st.session_state.ana_target_code = code_input
        
        k_labels = [config.K_TYPE_MAP[k]['name'] for k in config.K_TYPE_MAP.keys()]
        selected_k_idx = st.selectbox("主分析周期", range(len(k_labels)), format_func=lambda x: k_labels[x], index=0)
        selected_k_type = list(config.K_TYPE_MAP.keys())[selected_k_idx]
        selected_k_name = k_labels[selected_k_idx]
        
        adjust_type = st.selectbox("复权", ["前复权", "不复权", "后复权"], index=0)

    # 2. 模型参数
    saved_params = user_config.get("kronos_params", {})
    with st.expander("2. Kronos 模型参数", expanded=False):
        lookback = st.number_input("Lookback", 50, 500, saved_params.get("lookback", 100))
        pred_len = st.slider("步长", 5, 60, saved_params.get("pred_len", 10))
        n_preds = st.slider("采样", 1, 50, saved_params.get("n_preds", 10))

    # 3. AI 配置
    last_provider = user_config.get("last_provider", "Google Gemini (官方SDK)")
    with st.expander("3. AI 配置", expanded=True):
        data_src = st.radio("数据源", ["efinance", "akshare"], horizontal=True)
        force_sync = st.checkbox("强制云端同步")
        st.divider()
        
        llm_providers = list(user_config["providers"].keys())
        idx = llm_providers.index(last_provider) if last_provider in llm_providers else 0
        llm_provider = st.selectbox("AI 提供商", llm_providers, index=idx)
        
        p_config = user_config["providers"].get(llm_provider, {})
        api_key_input = st.text_input("API Key", value=p_config.get("api_key", ""), type="password")
        base_url_input = st.text_input("Base URL (可选)", value=p_config.get("base_url", ""))
        
        use_proxy = st.checkbox("启用代理", value=p_config.get("use_proxy", False))
        proxy_url = st.text_input("代理地址", value=p_config.get("proxy_url", "http://127.0.0.1:7890"))

        curr_model = p_config.get("model", "gemini-1.5-flash")
        all_models = list(set([curr_model] + st.session_state.model_list))
        all_models.sort()
        sel_model = st.selectbox("模型", all_models, index=all_models.index(curr_model) if curr_model in all_models else 0)
        
        if st.button("🔍 测试连接"):
            if api_key_input:
                with st.spinner("连接..."):
                    mods = get_available_models(llm_provider, api_key_input, base_url_input, proxy_url if use_proxy else None)
                    if mods: st.session_state.model_list = mods; st.success("成功")
                    else: st.error("失败")

        st.markdown("---")
        st.caption("AI 上下文偏好")
        saved_ctx = user_config.get("ai_context", {})
        ctx_news = st.checkbox("包含新闻", value=saved_ctx.get("news", True))
        ctx_kronos = st.checkbox("包含 Kronos 主图数据", value=saved_ctx.get("kronos_main", True)) # 注意 key
        ctx_tech = st.checkbox("包含技术指标", value=saved_ctx.get("tech", True))
        
        st.caption("多周期矩阵推理:")
        saved_frames = saved_ctx.get("kronos_frames", ["101"])
        selected_frames = st.multiselect("选择要投喂的周期", options=list(K_MAP.keys()), 
                                        format_func=lambda x: K_MAP[x], default=saved_frames)

    if st.button("💾 保存配置"):
        p_data = {"api_key": api_key_input, "base_url": base_url_input, "model": sel_model, "use_proxy": use_proxy, "proxy_url": proxy_url}
        kronos_p = {"lookback": lookback, "pred_len": pred_len, "n_preds": n_preds}
        ai_ctx = {"news": ctx_news, "tech": ctx_tech, "kronos_main": ctx_kronos, "kronos_frames": selected_frames}
        ConfigManager.save_config(llm_provider, p_data, ai_ctx, kronos_p)
        st.success("已保存")

# ================== 主界面 ==================
tab_screener, tab_analysis = st.tabs(["🔍 离线选股器", "📈 深度融合台"])

# 1. 选股器
with tab_screener:
    st.markdown("### ⚡ 本地极速筛选")
    df_local, file_time = StockDataProvider.get_market_snapshot_local()
    if df_local.empty: st.warning("本地数据为空，请运行 `python market_updater.py`。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        pe_min, pe_max = c1.slider("PE范围", 0, 200, (0, 60))
        cap_min = c2.number_input("市值(亿)", 0, value=30)
        chg_min = c3.slider("涨幅%", -10.0, 10.0, -3.0)
        all_inds = ['全部'] + list(df_local['所处行业'].dropna().unique())
        sel_ind = c4.selectbox("行业", all_inds)
        mask = (df_local['动态市盈率'] >= pe_min) & (df_local['动态市盈率'] <= pe_max) & (df_local['总市值'] >= cap_min * 100000000) & (df_local['涨跌幅'] >= chg_min)
        if sel_ind != '全部': mask = mask & (df_local['所处行业'] == sel_ind)
        res_df = df_local[mask].copy()
        st.success(f"筛选结果: {len(res_df)} 只")
        st.dataframe(res_df, width="stretch", hide_index=True, column_config={"代码": st.column_config.TextColumn("代码"), "涨跌幅": st.column_config.NumberColumn("涨跌幅", format="%.2f %%")})
        st.caption("提示：请手动复制感兴趣的股票代码到【深度融合台】进行分析。")

# 2. 深度融合台
with tab_analysis:
    code = st.session_state.ana_target_code
    real_info = StockDataProvider.get_realtime_info([code])
    name = real_info.loc[code].get('股票名称', '未知') if not real_info.empty and code in real_info.index else "未知"
    price = real_info.loc[code].get('最新价', 0) if not real_info.empty and code in real_info.index else 0

    st.markdown(f"### 🧬 {name} (`{code}`)  ¥{price}  <span style='font-size:0.6em; color:gray'>| {selected_k_name}</span>", unsafe_allow_html=True)

    if st.button("🚀 启动主图分析", type="primary", use_container_width=True):
        with st.status("正在分析...", expanded=True) as status:
            st.write(f"1. 获取 {selected_k_name} 数据...")
            csv_path = repo.get_kline(code, k_type=selected_k_type, source=data_src, adjust=adjust_type)
            
            if csv_path:
                st.write("2. 运行主策略引擎...")
                df = pd.read_csv(csv_path)
                
                predictor = StockPredictor(data_file=csv_path, output_dir='./output', plot_file='./output/future.png', 
                                           n_predictions=n_preds, lookback=lookback, pred_len=pred_len, 
                                           stock_code=code, verbose=False, save_details=True)
                kronos_res = predictor.run_analysis()
                rise_prob = kronos_res['statistics']['close'].get('rise_probability', 0.5) if kronos_res else 0.5
                
                engine = StrategyEngine(df)
                res = engine.run_analysis(kronos_rise_prob=rise_prob)
                
                st.session_state.step3_strategy = res
                st.session_state.step3_strategy['kronos_main'] = kronos_res 
                st.session_state.current_k_path = csv_path
                status.update(label="主分析完成", state="complete", expanded=False)
            else:
                status.update(label="数据失败", state="error")

    if st.session_state.step3_strategy:
        res = st.session_state.step3_strategy
        kronos_res = st.session_state.step3_strategy['kronos_main']
        rise_prob = kronos_res['statistics']['close'].get('rise_probability', 0.5) if kronos_res else 0.5
        signals = res['signals']
        
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("主周期看涨", f"{rise_prob*100:.1f}%", delta=f"{(rise_prob-0.5)*100:.1f}%")
        c2.metric("综合评分", f"{res['final_score']:.2f}")
        mom_val = signals.get('Momentum(KDJ/MACD)', 0)
        c3.metric("动量", f"{mom_val:.2f}", delta=f"{mom_val:.2f}")
        act_color = "green" if "买" in res['decision']['action'] else "red"
        c4.markdown(f"#### 📢 :{act_color}[{res['decision']['action']}]")
        
        with st.expander("📊 查看 Kronos 详细统计报告", expanded=False):
            if 'report_text' in kronos_res: st.text(kronos_res['report_text'])
            else: st.info("未生成文本报告")

        t1, t2 = st.tabs(["📈 技术全景", "🔮 预测轨迹"])
        with t1: st.plotly_chart(plot_tech_chart(res['processed_df'], res['key_levels']), use_container_width=True)
        with t2: 
            if os.path.exists('./output/future.png'): st.image('./output/future.png')

        st.divider()
        st.subheader(f"🤖 {llm_provider} 矩阵分析")
        
        if api_key_input:
            if st.button("🧠 生成深度研报"):
                apply_proxy(proxy_url if use_proxy else None)
                
                with st.spinner("AI 思考中 (含多周期矩阵推理)..."):
                    f_info = StockDataProvider.get_fundamentals(code)
                    advisor = LLMAdvisor(api_key_input, provider=llm_provider, model_name=sel_model, base_url=base_url_input)
                    
                    ctx_list = []
                    if ctx_news: 
                        news = StockDataProvider.get_stock_news(code, 5)
                        ctx_list.append(f"【资讯面】\n{news}")
                    
                    if ctx_tech:
                        ctx_list.append(f"【主周期技术 ({selected_k_name})】\n周期: {selected_k_name}\n动量: {mom_val:.2f}\n状态: {res['regime']['trend']}")
                    
                    k_stats = {}
                    if ctx_kronos and kronos_res:
                        k_stats = kronos_res['statistics']['close']
                        ctx_list.append(f"【主周期预测】\n波动系数: {k_stats.get('volatility_amplification', 0):.2f}\n上涨概率: {rise_prob*100:.1f}%")

                    # --- [核心修复] 多周期矩阵推理 ---
                    if selected_frames:
                        matrix_ctx = ["【多周期共振矩阵】"]
                        # 使用侧边栏配置的参数
                        run_params = {"n_preds": n_preds, "lookback": lookback, "pred_len": pred_len}
                        for frame in selected_frames:
                            # 简单的跳过主周期检查 (String Comparison)
                            if frame == selected_k_type: continue
                            
                            frame_name = K_MAP[frame]
                            sub_stats = run_silent_kronos(code, frame, data_src, adjust_type, run_params)
                            if sub_stats:
                                prob = sub_stats.get('rise_probability', 0.5)
                                matrix_ctx.append(f"- {frame_name}: 看涨概率 {prob*100:.1f}%")
                        
                        if len(matrix_ctx) > 1:
                            ctx_list.append("\n".join(matrix_ctx))

                    full_ctx = "\n\n".join(ctx_list)
                    df_raw = pd.read_csv(st.session_state.current_k_path)
                    
                    advice = advisor.get_advice(code, name, df_raw, k_stats, f_info, full_ctx, selected_k_name)
                    st.markdown(advice)
        else:
            st.info("配置 API Key 后解锁 AI 功能。")