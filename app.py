import streamlit as st
import pandas as pd
import json
import time
import io
import os
import ssl
import certifi
from openai import OpenAI

# ================= 0. 环境自愈逻辑 (针对 Mac SSL 修复) =================
try:
    os.environ['SSL_CERT_FILE'] = certifi.where()
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

# ================= 1. 系统配置 =================
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# ================= 2. 系统提示词 (严格遵循原文，未动一字) =================
SYSTEM_PROMPT = """请将以下内容分类为：数学、物理、化学、生物。如果不是学科题目或内容不完整，输出‘其他’。
【核心判据：主导领域原则 (Primary Domain Principle)】
请判断题目最终是为了解决哪个领域的问题，而非仅看使用了什么工具。
主体优先： 依据题目中主要研究实体或现象所属的学科进行分类。
工具剥离： 如果题目引用了其他学科的公式、定律或计算方法作为工具，来解释当前研究对象的性质，请忽略 these 工具的学科属性。
逻辑示例： 用 B 学科的方法解决 A 学科的问题 归类为 A 学科。
【形式服从目标】：忽略编程语法或抽象符号的表现形式，以最终考核的任务目标为准。
若交付是数值、公式证明或计算结论，归类为数学/物理等（如：计算面积）。
若交付是程序实现、代码逻辑或形式系统描述，归类为其他（如：编写函数、定义逻辑系统）。

信心分： 范围 0.0 - 1.0。
0.9 - 1.0： 术语极其明确，学科边界清晰。
0.7 - 0.8： 存在少量跨学科背景，但主导学科明显。
0.5 - 0.6： 典型的边缘/交叉学科，判定存在一定主观性。

请仅以 JSON 格式输出结果：
{
  "subject": "分类结果",
  "reason": "简短理由",
  "confidence": 0.95
}"""

# ================= 3. 核心处理逻辑 =================

def get_prediction(text, client):
    """单条数据处理及 API 调用"""
    if pd.isna(text) or str(text).strip() == "":
        return "其他", "内容为空", 1.0

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"请对以下内容进行学科分类：\n{str(text)}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            res = json.loads(completion.choices[0].message.content)
            return (
                res.get("subject"),
                res.get("reason"),
                res.get("confidence")
            )
        except Exception:
            if attempt < 2:
                time.sleep(1.5)
                continue
            return "Error", "API或解析异常", 0.0

# ================= 4. UI 界面布局 (Streamlit 实现) =================

# 页面基础配置
st.set_page_config(page_title="学科分类专家系统", page_icon="🔬", layout="wide")

# 自定义 CSS 样式 (模拟深靛蓝学术风)
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3em; background-color: #4f46e5; color: white; }
    .stButton>button:hover { background-color: #4338ca; border: none; }
    h1 { color: #4f46e5; text-align: center; font-weight: 700; }
    .status-box { padding: 20px; border-radius: 10px; background-color: #ffffff; border: 1px solid #e2e8f0; }
    </style>
    """, unsafe_allow_html=True)

# 标题栏
st.markdown("<h1>🔬 学科分类专家系统</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #475569; font-size: 1.1em;'>基于“主导领域原则”与“形式服从目标”的工业级自动化分类平台</p>", unsafe_allow_html=True)

# 核心分类判据 (SOP) 折叠栏
with st.expander("📜 核心分类判据 (SOP)"):
    st.markdown(SYSTEM_PROMPT.split('请仅以 JSON')[0])

# 主界面两栏布局
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("### 📥 任务参数配置")
    api_input = st.text_input(
        "DeepSeek API Key", 
        type="password", 
        placeholder="在此粘贴您的 API Key...",
        help="您的密钥仅用于本次会话加密传输，不会被服务器持久化存储。"
    )
    file_input = st.file_uploader("上传待分类 Excel 文件", type=["xlsx", "xls"])
    run_btn = st.button("🚀 开启智能学科分类")

with col2:
    st.markdown("### 📤 处理结果反馈")
    status_placeholder = st.empty()
    status_placeholder.text_area("运行状态实时监测", placeholder="等待任务指令...", height=100, disabled=True)
    download_placeholder = st.empty()

# ================= 5. 执行引擎 =================

if run_btn:
    if not api_input:
        st.error("❌ 请输入有效的 API Key")
    elif file_input is None:
        st.error("❌ 请上传待处理的 Excel 文件")
    else:
        try:
            # A. 环境准备
            client = OpenAI(api_key=api_input, base_url=BASE_URL)
            df = pd.read_excel(file_input)
            
            # B. 自动定位目标列
            blacklist = ['结果', 'label', 'reason', 'score', '信心分', '分类', '预测', '学科', 'subject']
            text_cols = [c for c in df.select_dtypes(include=['object']).columns if not any(b in str(c).lower() for b in blacklist)]
            
            if not text_cols:
                st.error("❌ 未能在表格中识别到有效的文本列")
            else:
                target_col = df[text_cols].apply(lambda x: x.astype(str).str.len()).mean().idxmax()
                status_placeholder.info(f"✅ 已自动识别目标列：【{target_col}】\n正在初始化处理...")

                # C. 批量分类处理
                results = []
                total = len(df)
                progress_bar = st.progress(0)
                
                for i, text in enumerate(df[target_col]):
                    res = get_prediction(text, client)
                    results.append(res)
                    # 更新进度
                    percent = (i + 1) / total
                    progress_bar.progress(percent, text=f"智能分类中... {i+1}/{total}")
                
                # D. 合并结果并准备导出
                res_df = pd.DataFrame(results, columns=['分类结果', '原因分析', '信心分'])
                final_df = pd.concat([df, res_df], axis=1)
                
                # 将 Excel 写入二进制流供下载
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    final_df.to_excel(writer, index=False)
                
                status_placeholder.success(f"✅ 处理完成！已处理 {total} 条数据。")
                
                # 启用下载按钮
                download_placeholder.download_button(
                    label="📥 点击下载分类审计报告",
                    data=output.getvalue(),
                    file_name="Subject_Classification_Final.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        except Exception as e:
            st.error(f"发生意外错误: {str(e)}")
