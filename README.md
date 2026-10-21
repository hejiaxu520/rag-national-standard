# 人工智能国家标准智能问答系统（RAG）

基于检索增强生成（RAG）技术的国家标准智能问答系统，可将 PDF 标准文档解析、向量化存储，并通过大语言模型实现带原文出处的精准问答。

## 功能特性

- **PDF 结构化解析**：采用 PyMuPDF4LLM 提取文本，保留真实页码与标题层级（Markdown），解析失败自动降级 pypdf。
- **语义检索**：基于 ChromaDB 向量数据库 + 豆包多模态嵌入，按语义相似度而非关键词匹配检索。
- **答案可溯源**：回答附带原文页码出处，杜绝大模型"编造"内容。
- **Web 问答界面**：基于 Gradio，浏览器内即开即用。

## 技术栈

| 环节 | 技术 |
|------|------|
| 框架 | LangChain |
| PDF 解析 | PyMuPDF4LLM / pypdf |
| 文本切分 | RecursiveCharacterTextSplitter |
| 向量化 | 豆包多模态 Embedding（火山方舟 API） |
| 向量数据库 | ChromaDB |
| 大语言模型 | 豆包（doubao-seed） |
| 界面 | Gradio |

## 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key（禁止硬编码，走环境变量）

```powershell
# PowerShell
$env:VOLC_API_KEY = "你的火山方舟ark密钥"
```

```bash
# Linux / macOS
export VOLC_API_KEY="你的火山方舟ark密钥"
```

可选环境变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `VOLC_API_KEY` | 火山方舟 API Key（必填） | - |
| `VOLC_BASE_URL` | 接口地址 | `https://ark.cn-beijing.volces.com/api/v3` |
| `VOLC_LLM_MODEL` | 生成模型 | `doubao-seed-2-0-mini-260428` |
| `VOLC_EMB_MODEL` | 嵌入模型 | - |

### 3. 使用

将 PDF 标准文档放入 `docs/` 目录，然后：

```bash
python rag_system_v2.py build              # 构建向量库
python rag_system_v2.py serve              # 启动 Web 问答界面
python rag_system_v2.py test "你的问题"     # 命令行单次问答验证
python rag_system_v2.py info               # 查看环境与向量库状态
```

启动后浏览器自动打开 `http://127.0.0.1:7860`，输入问题即可获取带页码出处的回答。

## 项目结构

```
├── rag_system_v2.py   # 主程序（解析、切分、向量化、检索、问答、Gradio 界面）
├── app.py             # V1 版本（PyPDFLoader 方案）
├── test_official.py   # 早期调试脚本
├── docs/              # 存放待解析的 PDF 标准文档
└── requirements.txt   # 依赖清单
```

## 说明

- 密钥通过 `os.environ.get("VOLC_API_KEY")` 读取，代码中不含任何硬编码密钥。
- 向量库与模型文件体积较大，已通过 `.gitignore` 排除，需在本机自行构建/下载。
