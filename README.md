🛡️ Kronos 终端 (V1.0)

Kronos 是一个基于 Python 和 Streamlit 构建的高级 A 股分析终端。它集成了全市场选股筛选器、深度 F10 基本面分析、资金流向监控、Kronos(默认使用CPU） 时序预测模型以及基于 LLM (大语言模型) 的智能研报生成功能。
不要问为什么默认不使用GPU,因为我没有@@，后面有修改方法。

V1.0 版本重点增强了数据获取的稳定性（多线程代理池、死磕补录机制）以及 UI 的响应速度（多级缓存）。
✨ 核心功能

    🔍 策略选股工厂：基于 PE/PB/ROE/股息率/RPS 等多维度的全市场实时筛选。

    📈 深度融合台：

        F10 深度档案：核心题材、主营业务、价值资讯（直连东方财富 DataCenter）。

        技术全景：K 线、均线、MACD、RSI 及支撑/阻力位分析。

        资金博弈：直连 Push2 接口的主力资金流向监控。

        Kronos 预测：基于深度学习的时序价格预测。

    🧠 AI 智能研报：集成 Gemini/DeepSeek/OpenAI，基于实时数据生成机构级研报。

    🛡️ 健壮的网络层：

        内置 代理池管理器 (Proxy Manager)，支持自动验证、冷却和复活。

        支持 多线程并发 + 智能补录，极速抓取全市场 5000+ 股票数据。

        直连/代理热切换：在深度分析界面可一键切换网络模式。

🛠️ 环境准备
1. 系统要求

    操作系统：Windows / macOS / Linux

    Python 版本：3.8 - 3.11 (推荐 3.10)

2. 获取代码

确保您的项目目录结构如下：
code Text

    
Kronos/

├── app.py                  # 主程序入口 (Streamlit)

├── market_updater.py       # 数据更新器 (多线程爬虫)

├── stock_data_provider.py  # 数据提供层 (DataCenter API)

├── data_layer.py           # K线数据层

├── proxy_manager.py        # 代理池管理器

├── config_manager.py       # 配置管理器

├── stock_predictor.py      # 预测模型 (需配合 model/ 文件夹)

├── strategy_engine.py      # 策略引擎

├── quant_engine.py         # 量化指标计算

├── config.py               # 静态配置

├── requirements.txt        # 依赖列表 (见下文)

└── data_repo/              # 数据存放目录 (自动生成)

    ├── daily/
    
    ├── market_snapshot_full.csv
    
    └── ...

  


🚀 部署与使用指南
第一步：初始化数据 (至关重要)

首次运行时，本地没有市场快照数据，选股器会报错。您需要运行更新器来抓取全市场数据。

方法 A：命令行运行 (推荐首次使用)
code Bash

    
python market_updater.py

####只影响选股工厂下的功能，不更新也不影响深度融合台使用

  

    选择 1 进行全量更新。

    程序会自动轮询代理池（如果配置了）或使用直连，抓取 A 股 5000+ 只股票的行情、财务和题材数据。

    等待显示 ✅ 快照更新成功！。

方法 B：在 UI 中运行
不建议使用UI界面更新，容易报错，最好在终端中使用market_updater.py脚本更新
启动 App 后，在侧边栏 "5. 💾 数据维护" 中点击 "🚀 全量更新"。
####只影响选股工厂下的功能，不更新也不影响深度融合台使用

第二步：启动终端

在终端中运行：
code Bash

    
streamlit run app.py

  

浏览器将自动打开 http://localhost:8501。
第三步：配置代理池 (可选但推荐)

为了防止高频访问被封 IP，建议配置代理池。

    在左侧侧边栏展开 "4. 🌐 网络与代理池"。

    在文本框中粘贴代理 IP（格式：http://ip:port，每行一个，目前配置文件中包含了3000+的代理池，默认即可）。

    点击 "➕ 添加至代理池"。

    程序会自动管理这些代理的生命周期。

第四步：配置 AI 模型

    在左侧侧边栏展开 "3. AI 配置"。

    选择提供商 (如 Google Gemini, DeepSeek)。

    输入对应的 API Key。（aistudio.google 申请即可，没测试过deepseek,如果默认的不能使用，修改接口地址即可）

    点击 "💾 保存配置"。

📖 使用手册
1. 策略选股工厂

    载入：启动时会自动加载本地缓存的快照数据（速度极快）。

    筛选：使用滑块调整 PE、ROE、市值等参数，或点击上方的一键策略按钮（如“💰 高息红利”）。

    选中：在表格中点击任意一行股票，系统会自动锁定该股票代码。

   ######必须update一次才可以正常使用

3. 深度融合台

切换到 "📈 深度融合台" 标签页：

    实时数据：顶部显示实时价格（默认直连，速度快）。

    代理开关：如果发现数据一直加载失败或显示 N/A，请打开右上角的 "🌍 启用数据代理" 开关。

    深度 F10：向下滚动查看 "📖 深度 F10 档案"，包含核心题材、主营业务和价值资讯。

    AI 研报：点击 "🧠 生成深度研报"，AI 将综合页面上的所有数据为您生成分析报告。

    ######不需要update也可以正常使用，左侧输入股票代码即可

❓ 常见问题排查

#Q1: 启动后报错 FileNotFoundError: market_snapshot_full.csv

    解决：请先运行 python market_updater.py 完成第一次数据初始化。

#Q2: 深度融合台显示 "暂无题材数据" 或 "N/A"

    原因：本地 IP 可能被东方财富暂时限制，或者该股票为次新股/特殊标的。

    解决：

        打开界面上的 "🌍 启用数据代理" 开关。

        检查 stock_data_provider.py 是否为 V1.0 版本（支持混合接口）。

#Q3: 输入股票代码时界面卡顿

    解决：这是 Streamlit 的特性。请确保一次性输入完整的 6 位代码并回车。V1.0 版本已加入防抖逻辑，只有输入满 6 位数字时才会触发刷新。

#Q4: 代理池全部显示 "冷却中"

    解决：说明您添加的代理质量较差或已失效。

        在侧边栏点击 "♻️ 恢复冷却代理" 强行重试。

        或者清空代理池，添加新的高质量代理。

        如果没有好代理，请保持代理池为空，程序会自动回退到 本机直连 模式（Market Updater 会自动轮询域名，通常也能成功）。

#Q5：Kronos无法下载 
    解决：国内通过huggingface下载，通过以下方式可以成功
        模型目录是：Kronos/NeoQuasar

    conda 虚拟环境下：
    export HF_ENDPOINT="https://hf-mirror.com"
    mkdir -p ./kronos/NeoQuasar/Kronos-base
    mkdir -p ./kronos/NeoQuasar/Kronos-mini
    mkdir -p ./kronos/NeoQuasar/Kronos-small
    mkdir -p ./kronos/NeoQuasar/Kronos-Tokenizer-2k
    mkdir -p ./kronos/NeoQuasar/Kronos-Tokenizer-base

    将以下内容写入py脚本，然后执行，注意要提前创建目录
    
    from huggingface_hub import snapshot_download

     #Models
    snapshot_download(repo_id="NeoQuasar/Kronos-base", local_dir="./kronos/NeoQuasar/Kronos-base")
    snapshot_download(repo_id="NeoQuasar/Kronos-mini", local_dir="./kronos/NeoQuasar/Kronos-mini")
    snapshot_download(repo_id="NeoQuasar/Kronos-small", local_dir="./kronos/NeoQuasar/Kronos-small")

     #Tokenizers
    snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-2k", local_dir="./kronos/NeoQuasar/Kronos-Tokenizer-2k")
    snapshot_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", local_dir="./kronos/NeoQuasar/Kronos-Tokenizer-base")

    print("Raw model files downloaded successfully!")


#Q6：怎样使用GPU
    解决：修改stock_predictor.py第85行将cpu改为cuda:0
    修改前
    self.predictor = KronosPredictor(self.model, self.tokenizer, device="cpu", max_context=512)
    修改后
    self.predictor = KronosPredictor(self.model, self.tokenizer, device="cuda:0", max_context=512)


⚠️ 免责声明

本项目仅供计算机编程学习和金融数据分析研究使用。

    数据来源：所有数据均来自公开网络接口（东方财富、新浪财经等），本项目不保证数据的准确性、及时性和完整性。

    投资风险：项目生成的任何分析、预测或研报不构成任何投资建议。股市有风险，入市需谨慎。开发者不对任何投资亏损负责。
