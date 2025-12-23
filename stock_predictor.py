import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import matplotlib.font_manager as fm
from datetime import datetime
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

# 路径修正
sys.path.append("../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ImportError:
    class FakeKronosPredictor:
        def __init__(self, *args, **kwargs): pass
        def predict(self, *args, **kwargs): raise ImportError("Kronos 模型未加载")
    KronosPredictor = FakeKronosPredictor

class StockPredictor:
    # --- 统计量配置 (从 ML 版移植) ---
    STAT_COLORS = {
        'mean': 'red', 'median': 'orange', 'std': 'green', 
        'q1': 'purple', 'q3': 'brown', 'p10': 'coral', 'p90': 'lime',
        'min': 'darkred', 'max': 'darkgreen'
    }

    def __init__(self, data_file=None, output_dir=None, plot_file=None,
                 n_predictions=5, lookback=300, pred_len=30, validation_len=0,
                 adjust_lookback_ratio=None, 
                 # 默认开启全统计量
                 selected_stats=['mean', 'median', 'std', 'q1', 'q3', 'p10', 'p90', 'min', 'max', 'skewness', 'kurtosis'],
                 save_details=False,
                 font_file='/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
                 stock_code=None, verbose=True):
        self.data_file = data_file
        self.output_dir = output_dir or './output/'
        self.plot_file = plot_file
        self.n_predictions = n_predictions
        self.lookback = lookback
        self.pred_len = pred_len
        self.validation_len = validation_len
        self.adjust_lookback_ratio = adjust_lookback_ratio
        self.selected_stats = selected_stats
        self.save_details = save_details
        self.stock_code = stock_code
        self.font_file = font_file
        self.verbose = verbose
        
        # --- 字体智能初始化 (保留 V16 的强力逻辑) ---
        self.custom_font = self._find_chinese_font()
        plt.rcParams['axes.unicode_minus'] = False
        
        self.model = None
        self.predictor = None

    def _find_chinese_font(self):
        """自动寻找可用的中文字体"""
        if self.font_file and os.path.exists(self.font_file):
            try: return fm.FontProperties(fname=self.font_file)
            except: pass

        candidate_paths = [
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
            '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
            '/System/Library/Fonts/PingFang.ttc', 'C:\\Windows\\Fonts\\msyh.ttc'
        ]
        for path in candidate_paths:
            if os.path.exists(path):
                try:
                    fm.fontManager.addfont(path)
                    p = fm.FontProperties(fname=path)
                    plt.rcParams['font.family'] = p.get_name()
                    return p
                except: continue
        return None

    def load_model(self):
        if not self.model:
            self.tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", cache_dir="NeoQuasar", local_files_only=True)
            self.model = Kronos.from_pretrained("NeoQuasar/Kronos-base", cache_dir="NeoQuasar", local_files_only=True)
            self.predictor = KronosPredictor(self.model, self.tokenizer, device="cpu", max_context=512)

    def load_data(self):
        if not os.path.exists(self.data_file): raise FileNotFoundError(f"{self.data_file} 不存在")
        df = pd.read_csv(self.data_file)
        if 'volume' not in df.columns: df['volume'] = 0
        if 'timestamps' not in df.columns:
            if 'date' in df.columns: df.rename(columns={'date': 'timestamps'}, inplace=True)
            else: raise ValueError("缺少 timestamps 列")
        
        df['timestamps'] = pd.to_datetime(df['timestamps'], errors='coerce')
        df.dropna(subset=['timestamps'], inplace=True)
        df = df.sort_values('timestamps').reset_index(drop=True)

        y_true_df = None
        if self.validation_len > 0 and len(df) > self.validation_len:
            actual_val_len = min(self.validation_len, len(df))
            x_df_full = df.iloc[:-actual_val_len].copy()
            y_true_df = df.iloc[-actual_val_len:].copy()
            y_true_df = y_true_df.iloc[:self.pred_len]
            self.pred_len = len(y_true_df)
        else:
            x_df_full = df.copy()
        
        historical_df = x_df_full.copy()
        x_df = historical_df.tail(self.lookback)
        x_timestamp = historical_df.tail(self.lookback)['timestamps']

        last_ts = historical_df['timestamps'].iloc[-1]
        delta = pd.Timedelta(minutes=5)
        if len(historical_df) > 1:
            diffs = historical_df['timestamps'].diff().dropna()
            if not diffs.empty: delta = diffs.mode()[0]
        
        if y_true_df is not None: y_timestamp = y_true_df['timestamps']
        else: y_timestamp = pd.date_range(start=last_ts + delta, periods=self.pred_len, freq=delta)
        return x_df, x_timestamp, y_timestamp, historical_df, y_true_df

    def run_predictions(self, x_df, x_timestamp, y_timestamp):
        all_preds = []
        y_ts_series = pd.Series(y_timestamp).reset_index(drop=True)
        for i in range(self.n_predictions):
            try:
                pred_df = self.predictor.predict(df=x_df, x_timestamp=x_timestamp, y_timestamp=y_ts_series, pred_len=self.pred_len, T=0.7, top_p=0.9, sample_count=3, verbose=False)
                pred_df['timestamps'] = y_timestamp
                all_preds.append(pred_df)
            except Exception as e: print(f"预测失败: {e}")
        return all_preds

    # --- 增强版统计计算 ---
    def calculate_statistics(self, pred_dfs, y_true_df, historical_df):
        if not pred_dfs: return {'close': {}}
        closes = np.array([df['close'].values for df in pred_dfs])
        
        # 基础统计
        stats_res = {
            'mean': np.mean(closes, axis=0),
            'median': np.median(closes, axis=0),
            'std': np.std(closes, axis=0),
            'min': np.min(closes, axis=0),
            'max': np.max(closes, axis=0),
            'q1': np.percentile(closes, 25, axis=0),
            'q3': np.percentile(closes, 75, axis=0),
            'p10': np.percentile(closes, 10, axis=0),
            'p90': np.percentile(closes, 90, axis=0),
            'skewness': stats.skew(closes, axis=0),
            'kurtosis': stats.kurtosis(closes, axis=0)
        }
        
        # 转换为 Series
        time_idx = pred_dfs[0]['timestamps']
        result = {k: pd.Series(v, index=time_idx) for k,v in stats_res.items()}
        
        # 核心标量指标
        last_hist = historical_df['close'].iloc[-1]
        rise_prob = np.mean(closes[:, -1] > last_hist)
        
        hist_vol = historical_df['close'].pct_change().std()
        pred_vol = pd.Series(stats_res['mean']).pct_change().std()
        vol_amp = pred_vol / hist_vol if hist_vol != 0 else 0
        
        result.update({
            'rise_probability': rise_prob,
            'fall_probability': 1 - rise_prob,
            'volatility_amplification': vol_amp
        })
        
        # RMSE (回测模式)
        if y_true_df is not None:
            y_true = y_true_df['close'].values
            # 确保长度一致
            valid_len = min(len(y_true), len(stats_res['mean']))
            rmse = np.sqrt(mean_squared_error(y_true[:valid_len], stats_res['mean'][:valid_len]))
            result['rmse'] = rmse
            
        return {'close': result}

    # --- 移植自 ML 版的文本报告生成 ---
    def print_statistics_report(self, stats_results):
        """生成详细的文本报告"""
        lines = []
        def log(s=""): lines.append(s)
        
        log(f"📊 Kronos 统计分析报告 - {self.stock_code or 'Unknown'}")
        log("=" * 50)
        
        stats = stats_results.get('close', {})
        
        # 1. 核心标量
        log("【核心预测指标】")
        if 'rise_probability' in stats:
            log(f"  📈 上涨概率: {stats['rise_probability']*100:.2f}%")
        if 'volatility_amplification' in stats:
            log(f"  🌊 波动放大: {stats['volatility_amplification']:.2f}x (vs历史)")
        if 'rmse' in stats and not pd.isna(stats['rmse']):
            log(f"  📉 预测误差(RMSE): {stats['rmse']:.4f}")
            
        # 2. 终点分布统计
        log("\n【预测终点分布统计】")
        display_keys = {
            'mean': '均值', 'median': '中位数', 'std': '标准差',
            'min': '最小值', 'max': '最大值',
            'q1': 'Q1(25%)', 'q3': 'Q3(75%)',
            'skewness': '偏度', 'kurtosis': '峰度'
        }
        
        for key, name in display_keys.items():
            if key in stats and isinstance(stats[key], pd.Series):
                val = stats[key].iloc[-1]
                log(f"  {name:<10}: {val:.4f}")
                
        return "\n".join(lines)

    # --- 移植自 ML 版的文件保存 ---
    def save_detailed_results(self, pred_dfs, stats_results, timestamp_str, report_text):
        if not self.save_details: return
        
        base_name = f"{self.stock_code or 'UNK'}_{timestamp_str}"
        save_path = os.path.join(self.output_dir, base_name)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存统计 CSV
        if 'close' in stats_results:
            # 过滤掉标量，只保存序列
            df_stats = pd.DataFrame({k: v for k, v in stats_results['close'].items() if isinstance(v, pd.Series)})
            df_stats.to_csv(os.path.join(save_path, "statistics.csv"), index=True)
            
        # 保存报告 TXT
        with open(os.path.join(save_path, "report.txt"), 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        # 保存每次预测
        for i, df in enumerate(pred_dfs):
            df.to_csv(os.path.join(save_path, f"sim_{i+1}.csv"), index=False)
            
        if self.verbose: print(f"✅ 详细数据已保存至: {save_path}")

    # --- 增强版绘图 (融合 V16 和 ML版) ---
    def create_plots(self, historical_df, pred_dfs, stats_results, y_true_df=None):
        if not pred_dfs: return
        
        # 数据准备
        plot_lookback = int(self.pred_len * self.adjust_lookback_ratio) if self.adjust_lookback_ratio else self.lookback
        hist_plot = historical_df.tail(plot_lookback).copy()
        total_len = len(hist_plot) + self.pred_len
        x_idx = np.arange(total_len)
        x_pred = x_idx[len(hist_plot):]
        
        # 字体字典
        t_dict = {'fontproperties': self.custom_font, 'fontsize': 14} if self.custom_font else {}
        l_dict = {'fontproperties': self.custom_font} if self.custom_font else {}
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1.5]})
        
        title_prefix = f"[{self.stock_code}] " if self.stock_code else ""
        ax1.set_title(f'{title_prefix}Kronos 趋势预测 (含成交量)', **t_dict)

        # === 子图 1: 价格 ===
        ax1.plot(x_idx[:len(hist_plot)], hist_plot['close'], color='blue', label='历史')
        
        # 绘制所有路径 (淡色)
        for df in pred_dfs:
            y_vals = np.concatenate([[hist_plot['close'].iloc[-1]], df['close'].values])
            x_vals = np.concatenate([[x_idx[len(hist_plot)-1]], x_pred])
            ax1.plot(x_vals, y_vals, color='cyan', alpha=0.1, linewidth=0.5)
        
        # 均值线
        stats = stats_results['close']
        if 'mean' in stats:
            mean_vals = stats['mean'].values
            y_mean = np.concatenate([[hist_plot['close'].iloc[-1]], mean_vals])
            x_mean = np.concatenate([[x_idx[len(hist_plot)-1]], x_pred])
            ax1.plot(x_mean, y_mean, color='red', linewidth=2, label='预测均值')

        if y_true_df is not None:
            y_true = np.concatenate([[hist_plot['close'].iloc[-1]], y_true_df['close'].values])
            ax1.plot(x_mean, y_true, color='black', linewidth=2, label='实际')

        ax1.set_ylabel('价格 (Price)', **l_dict)
        leg = ax1.legend(loc='upper left')
        if self.custom_font: plt.setp(leg.get_texts(), fontproperties=self.custom_font)
        ax1.grid(True, alpha=0.3)

        # === 子图 2: 多重置信区间 (新增功能) ===
        ax2.set_title('概率分布范围 (Probability Ranges)', **t_dict)
        ax2.plot(x_idx[:len(hist_plot)], hist_plot['close'], color='blue', alpha=0.5)
        ax2.plot(x_mean, y_mean, color='red', linestyle='--')
        
        # 1. Min-Max (最宽, 灰色)
        if 'min' in stats and 'max' in stats:
            y_min = np.concatenate([[hist_plot['close'].iloc[-1]], stats['min'].values])
            y_max = np.concatenate([[hist_plot['close'].iloc[-1]], stats['max'].values])
            ax2.fill_between(x_mean, y_min, y_max, color='gray', alpha=0.1, label='Min-Max')
            
        # 2. P10-P90 (宽区间, 蓝色)
        if 'p10' in stats and 'p90' in stats:
            y_p10 = np.concatenate([[hist_plot['close'].iloc[-1]], stats['p10'].values])
            y_p90 = np.concatenate([[hist_plot['close'].iloc[-1]], stats['p90'].values])
            ax2.fill_between(x_mean, y_p10, y_p90, color='blue', alpha=0.1, label='P10-P90')
            
        # 3. Q1-Q3 (核心区间, 橙色)
        if 'q1' in stats and 'q3' in stats:
            y_q1 = np.concatenate([[hist_plot['close'].iloc[-1]], stats['q1'].values])
            y_q3 = np.concatenate([[hist_plot['close'].iloc[-1]], stats['q3'].values])
            ax2.fill_between(x_mean, y_q1, y_q3, color='orange', alpha=0.2, label='IQR (25-75%)')

        ax2.set_ylabel('范围 (Range)', **l_dict)
        leg2 = ax2.legend(loc='upper left')
        if self.custom_font: plt.setp(leg2.get_texts(), fontproperties=self.custom_font)
        ax2.grid(True, alpha=0.3)

        # === 子图 3: 成交量 ===
        ax3.set_title('成交量 (Volume)', **t_dict)
        colors = ['red' if c >= o else 'green' for o, c in zip(hist_plot['open'], hist_plot['close'])]
        ax3.bar(x_idx[:len(hist_plot)], hist_plot['volume'], color=colors, width=0.8, alpha=0.7)
        
        if pred_dfs and 'volume' in pred_dfs[0]:
            vol_preds = np.array([df['volume'].values for df in pred_dfs])
            ax3.bar(x_idx[len(hist_plot):], np.mean(vol_preds, axis=0), color='gray', alpha=0.5, label='预测量')

        ax3.set_ylabel('量 (Volume)', **l_dict)
        
        # X轴
        all_ts = pd.concat([hist_plot['timestamps'], pred_dfs[0]['timestamps'] if pred_dfs else pd.Series()]).reset_index(drop=True)
        tick_pos = np.linspace(0, total_len-1, 8, dtype=int)
        tick_labels = []
        for pos in tick_pos:
            if pos < len(all_ts):
                val = all_ts.iloc[pos]
                tick_labels.append("" if pd.isna(val) else val.strftime('%m-%d %H:%M'))
            else: tick_labels.append("")
        
        ax3.set_xticks(tick_pos)
        ax3.set_xticklabels(tick_labels, rotation=30, ha='right', **l_dict)

        plt.tight_layout()
        try:
            os.makedirs(os.path.dirname(self.plot_file), exist_ok=True)
            plt.savefig(self.plot_file, bbox_inches='tight', dpi=100)
        except Exception as e: print(f"保存图片失败: {e}")
        plt.close()

    def run_analysis(self):
        self.load_model()
        try:
            x_df, x_ts, y_ts, hist_df, y_true = self.load_data()
            preds = self.run_predictions(x_df, x_ts, y_ts)
            stats = self.calculate_statistics(preds, y_true, hist_df)
            
            # 生成文本报告
            report_text = self.print_statistics_report(stats)
            
            # 绘图
            if self.plot_file:
                self.create_plots(hist_df, preds, stats, y_true)
                
            # 保存详细结果
            if self.save_details:
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.save_detailed_results(preds, stats, ts_str, report_text)
                
            return {'predictions': preds, 'statistics': stats, 'true_data': y_true, 'report_text': report_text}
        except Exception as e:
            print(f"分析流程中断: {e}")
            import traceback
            traceback.print_exc()
            return None