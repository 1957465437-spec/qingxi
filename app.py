import streamlit as st
import pandas as pd
import json
import time
import io
import os
import ssl
import certifi
from openai import OpenAI

# ================= 0. 环境自愈逻辑 (针对 Mac SSL 证书修复) =================
try:
    os.environ['SSL_CERT_FILE'] = certifi.where()
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

# ================= 1. 系统配置 =================
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# ================= 2. 深度审计提示词 (严格保持不变) =================
SYSTEM_PROMPT = """你是一名极度专业且严谨的数据审计专家。你的任务是判定输入文本是否为“逻辑完备且可用于教学考核”的标准题目。

### 一、 分类核心哲学
- **标准 (0)**：有完整的考核任务（包括陈述式指令），逻辑链条闭合。允许背景中存在无关噪音，只要不干扰题目主体的理解。
- **脏 (1)**：逻辑链条断裂、关键参数缺失、或是纯粹的非题目干扰。

### 二、 判定红线（出现以下情况必判为 1）

1. **逻辑空洞（致命残缺）**：
   - **已知量缺失**：文本中明确提到“已知”、“如图”、“如下表”，但后续没有具体的数值、描述或是数据。
   - **待求量缺失**：仅有背景陈述或公式罗列，没有任何提问或要求（例如：只有一段定义或定理，没有要求简析或评价）。
   - **语义截断**：文本在连词（因为、但是、如果）或公式中段突然中止，导致无法理解完整意图。

2. **纯粹非题目废料**：
   - 文本主体不是题目。例如：纯广告、纯代码段、纯系统日志（NaN、ID、[音频]）、或是完全无意义的字符堆砌。
   - **语境孤儿**：仅有一句无法独立成题的片段（如：“答案选A”、“第12题：”）。

3. **结果严重泄露**：
   - 题干中直接包含了详细的解析步骤或标准答案。

### 三、 豁免原则（以下情况必须判为 0）

1. **形式豁免**：
   - **陈述式指令**：以“简析”、“论述”、“说明”、“比较”、“阐述”等动词开头的陈述句，只要考核目标明确，严禁判定为不完整。
2. **噪音豁免**：
   - 只要题目主体逻辑闭合，开头或结尾粘连的水印、广告、版权声明、日期、流水号等，**一律视为可接受的干扰，判定为 0**。

### 四、 强制审计流程（内部思维链）

1. **任务扫描**：文本是否发出了“指令”？（问号或“简析/计算”等动词）。如果没有，判 1。
2. **逻辑自洽审计**：尝试梳理：已知量是什么？求什么？如果由于文字缺失（如：已知a=... 后面没了）导致无法解题，判 1。
3. **疑点利益归于标准**：如果题目逻辑是完整的，只是由于多了一些文字噪音，**坚决判 0**。

### 五、 输出要求

请仅输出 JSON 格式的结果：
- **label**: 整数。1 代表脏数据，0 代表标准数据。
- **reason**: 字符串。请使用：**[逻辑闭合]、[参数缺失]、[语义截断]、[纯噪音]、[包含解析]**。
- **confidence**: 浮点数（0.0 到 1.0）。判定标准如下：
  - 0.9 - 1.0： 术语极其明确，学科边界清晰。
  - 0.7 - 0.8： 存在少量跨学科背景，但主导学科明显。
  - 0.5 - 0.6： 典型的边缘/交叉学科，判定存在一定主观性。"""

# ================= 3. 核心处理逻辑 =================

def get_prediction(text, client):
    if pd.isna(text) or str(text).strip() == "":
        return 1, "[纯噪音] 内容为空", 1.0
    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"请审计以下内容：\n{str(text)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            res = json.loads(completion.choices[0].message.content)
            return (res.get("label"), res.get("reason"), res.get("confidence"))
        except Exception:
            if attempt < 2:
                time.sleep(1.5)
                continue
            return "Error", "API或解析异常", 0.0

# ================= 4. UI 界面优化 (Streamlit 实现) =================

st.set_page_config(page_title="脏数据筛选工具", page_icon="🔬", layout="wide")

# 自定义 CSS 模仿青色 (Teal) 主题
st.markdown("""
    <style>
    .stButton>button { width: 100%; border-radius: 8px; background-color: #0d9488; color: white; height: 3em; }
    .stButton>button:hover { background-color: #0f766e; border: none; }
    h1 { color: #0d9488; text-align: center; }
    .stProgress > div > div > div > div { background-color: #0d9488; }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>🔬 脏数据筛选工具</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.1em;'>基于大规模语言模型，提供工业级的数据逻辑合规性自动判定</p>", unsafe_allow_html=True)

with st.expander("📌 审计准则与使用须知"):
    st.markdown("""
    - **标准 (0)**: 逻辑闭合，考核目标明确，允许少量水印干扰。
    - **脏 (1)**: 存在语义截断、关键参数缺失或非题目废料。
    - **自动列识别**: 系统将自动分析表格，选取内容最丰富的文本列进行审计。
    """)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📥 任务输入")
    api_input = st.text_input("API Key", type="password", placeholder="在此粘贴您的 DeepSeek API Key...", help="您的密钥仅用于本次本地会话，不会被服务器存储。")
    file_input = st.file_uploader("上传数据源 (.xlsx / .xls)", type=["xlsx", "xls"])
    run_btn = st.button("🚀 开始执行智能筛选")

with col2:
    st.markdown("### 📤 处理结果")
    status_placeholder = st.empty()
    status_placeholder.text_area("运行状态反馈", placeholder="等待任务启动...", height=100, disabled=True)
    download_placeholder = st.empty()

# ================= 5. 执行逻辑 =================

if run_btn:
    if not api_input:
        st.error("❌ 请输入有效的 API Key")
    elif file_input is None:
        st.error("❌ 请上传待处理的 Excel 文件")
    else:
        # A. 准备工作
        client = OpenAI(api_key=api_input, base_url=BASE_URL)
        df = pd.read_excel(file_input)
        
        # B. 自动识别列
        blacklist = ['结果', 'label', 'reason', 'score', '信心分', '分类', '预测', 'response', 'ID', '序号', '学科']
        text_cols = [c for c in df.select_dtypes(include=['object']).columns if not any(b in str(c).lower() for b in blacklist)]
        
        if not text_cols:
            st.error("❌ 未能在表格中识别到文本列")
        else:
            target_col = df[text_cols].apply(lambda x: x.astype(str).str.len()).mean().idxmax()
            status_placeholder.info(f"✅ 正在处理目标列：【{target_col}】")
            
            # C. 批量审计
            results = []
            total = len(df)
            progress_bar = st.progress(0)
            
            for i, text in enumerate(df[target_col]):
                res = get_prediction(text, client)
                results.append(res)
                # 更新进度
                progress_bar.progress((i + 1) / total, text=f"深度审计中... {i+1}/{total}")
            
            # D. 生成结果
            res_df = pd.DataFrame(results, columns=['审计结果', '判定依据', '模型信心分'])
            final_df = pd.concat([df, res_df], axis=1)
            
            # 存储到二进制流以便下载
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                final_df.to_excel(writer, index=False)
            excel_data = output.getvalue()
            
            # E. 更新 UI 状态
            status_placeholder.success(f"✅ 审计完成！已自动识别并处理目标列：【{target_col}】")
            
            download_placeholder.download_button(
                label="📥 点击下载筛选报告",
                data=excel_data,
                file_name="audit_report_final.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
