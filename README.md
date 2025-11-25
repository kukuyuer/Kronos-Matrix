    
# 🌌 Kronos 量化决策终端

Kronos 是一个基于 Streamlit 的全栈量化分析系统，集成了离线选股、多模型融合策略（Kronos Transformer + 传统技术指标）以及 LLM（DeepSeek/Gemini）智能投顾功能。

## ✨ 核心功能

### 1. ⚡ 离线极速选股器
- 基于后台定时更新的全市场快照，实现毫秒级筛选。
- 支持按 **PE、市值、涨跌幅、行业板块** 进行多维过滤。
- 彻底解决 efinance/AkShare 在线筛选易超时的问题。

### 2. 🧬 深度融合策略台
- **多模型共识**：融合 Kronos 时序预测模型 + 趋势跟踪 + 均值回归策略。
- **动量分析**：内置 MACD、KDJ 动量评分系统。
- **多周期矩阵**：支持同时分析 5分钟、30分钟、日线数据，捕捉长短周期共振。

### 3. 🤖 AI 智能投顾
- 支持 **DeepSeek**、**Google Gemini**、**OpenAI** 等大模型。
- **多维投喂**：自动整理技术指标、Kronos 预测概率、个股新闻，生成专业的投资研报。
- **网络增强**：支持独立配置代理，解决国内服务器连接 Google API 的问题。

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt

  

2. 初始化数据

首次运行前，需要下载市场数据（建议挂在后台运行）：
code Bash

    
python market_updater.py
# 选择选项 1 (快照) 或 2 (全量K线)

  

3. 下载模型权重

请将训练好的 Kronos 模型权重文件夹 NeoQuasar 放入项目根目录。
(此处可填写你的模型下载链接，例如 HuggingFace 或 网盘)
4. 启动终端
code Bash

    
streamlit run app.py

  

⚙️ 配置说明

启动后在侧边栏配置 API Key。配置会自动保存到本地 user_config_v3.json。

    数据源: 支持 efinance (极速) 和 AkShare (稳定)。

    AI 设置: 支持自定义 Base URL 和 HTTP 代理。

📂 目录结构

    app.py: 前端入口

    quant_engine.py: 量化策略核心逻辑

    stock_predictor.py: Kronos 模型推理与绘图

    market_updater.py: 后台数据维护脚本

    data_repo/: 本地数据仓库 (自动生成)

⚠️ 免责声明

本项目仅供学习与研究使用，不构成任何投资建议。
