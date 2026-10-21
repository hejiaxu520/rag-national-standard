#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
人工智能国家标准智能问答系统 V2
================================
相对 V1(app.py) 的升级点：
  1. PDF 解析：PyPDFLoader -> PyMuPDF4LLM 结构化解析，保留标题层级(Markdown)与真实页码；
     解析失败时自动降级 pypdf，保证流程不中断。
     （原计划用 MinerU，但 magic-pdf 1.3.12 依赖的 ch_PP-OCRv3_det_infer.pth
       在 modelscope 与 hf-mirror 两个官方源均已下线，属上游版本错配，故改用
       Artifex 官方 RAG 库 PyMuPDF4LLM：零模型下载、CPU 秒级、页码准确。）
  2. 密钥安全：代码内不含任何硬编码 API Key，统一 os.environ.get("VOLC_API_KEY") 读取。
  3. 处理范围：仅处理 docs 下按文件名排序的前 MAX_PDF_COUNT 个 PDF，其余跳过。
  4. 页码准确：pymupdf4llm 的 metadata['page'] 为 1-based，直接作为引用页码，
     不可再 +1（V1 的 PyPDFLoader 是 0-based，此处口径不同，已显式处理）。

用法：
  python rag_system_v2.py build            # 构建（重建）向量库
  python rag_system_v2.py serve            # 启动 Gradio 问答界面
  python rag_system_v2.py test "你的问题"   # 单次问答验证，结果写入 verify_report.md
  python rag_system_v2.py info             # 打印向量库与解析器状态摘要

环境变量：
  VOLC_API_KEY      必填，火山方舟 API Key（禁止硬编码）
  VOLC_BASE_URL     可选，默认 https://ark.cn-beijing.volces.com/api/v3
  VOLC_LLM_MODEL    可选，默认 doubao-seed-2-0-mini-260428
  VOLC_EMB_MODEL    可选，默认 ep-20261018185559-ptlmc
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# 路径与全局配置
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, "docs")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db_v2")
REPORT_PATH = os.path.join(BASE_DIR, "verify_report.md")
ERROR_LOG_PATH = os.path.join(BASE_DIR, "error_log.txt")
SUMMARY_PATH = os.path.join(BASE_DIR, "build_summary.json")

MAX_PDF_COUNT = 3          # 红线：仅处理前 3 个 PDF
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4
MIN_PAGE_CHARS = 20        # 少于此字符数视为空白页/纯分隔符页，丢弃

# 红线：密钥只从环境变量读取，任何情况下不写入代码
API_KEY = os.environ.get("VOLC_API_KEY")
BASE_URL = os.environ.get("VOLC_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
LLM_MODEL = os.environ.get("VOLC_LLM_MODEL", "doubao-seed-2-0-mini-260428")
EMBEDDING_MODEL = os.environ.get("VOLC_EMB_MODEL", "ep-20261018185559-ptlmc")


def log(msg: str) -> None:
    """带时间戳的输出，便于长任务监控。"""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def append_error_log(stage: str, detail: str) -> None:
    try:
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} | {stage} =====\n{detail}\n")
    except Exception:
        pass


def require_api_key() -> str:
    """缺失密钥时按既定规则暂停并提示用户手动设置，绝不使用兜底硬编码值。"""
    key = os.environ.get("VOLC_API_KEY")
    if not key:
        msg = (
            "未检测到环境变量 VOLC_API_KEY，已按规则暂停。\n"
            "请在当前终端手动设置后重跑（密钥不会被写入任何文件）：\n"
            '  PowerShell:  $env:VOLC_API_KEY = "你的ark密钥"\n'
            '  CMD:         set VOLC_API_KEY=你的ark密钥\n'
            "设置完成后执行： python rag_system_v2.py build"
        )
        append_error_log("缺少 VOLC_API_KEY", msg)
        raise SystemExit(msg)
    return key


# ---------------------------------------------------------------------------
# PDF 解析
# ---------------------------------------------------------------------------
@dataclass
class ParseResult:
    source: str
    engine: str                                    # "pymupdf4llm" | "pypdf"
    text: str
    pages: List[Tuple[int, str]] = field(default_factory=list)   # (1-based页码, 该页文本)
    warning: str = ""


def clean_markdown(text: str) -> str:
    """
    清洗 pymupdf4llm 输出的 Markdown 噪声。
    实测国标 PDF 的典型噪声：
      - 文字被误加方括号：'[范围]'、'[发布]'
      - 分隔符插入空格：'GB [/] T 45288.1 [—] 2025'
      - 目次点线：'………………'
      - 中文字间空格：'人工智能 具身智能'（会把短语割裂，影响检索召回）
      - 分页横线：'-----'
    """
    t = text
    t = t.replace(" [/] ", "/").replace("[—]", "—").replace(" [—] ", "—")
    t = re.sub(r"\[([^\]\[]{1,80})\]", r"\1", t)          # 去掉误加的方括号
    t = re.sub(r"[·…]{3,}|\.{4,}", " ", t)                 # 目次点线
    t = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", t)   # 中文字间空格
    t = re.sub(r"^-{3,}\s*$", "", t, flags=re.M)           # 分页横线
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _parse_with_pymupdf4llm(pdf_path: str) -> ParseResult:
    """
    PyMuPDF4LLM 分页解析：返回 Markdown 文本并保留真实页码。
    注意 metadata['page'] 为 1-based，直接用作引用页码，禁止再 +1。
    """
    import pymupdf4llm

    raw_chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True, show_progress=False)
    pages: List[Tuple[int, str]] = []
    for c in raw_chunks:
        meta = c.get("metadata", {}) or {}
        page_no = meta.get("page", 0)
        text = clean_markdown(c.get("text", ""))
        if len(text) < MIN_PAGE_CHARS:      # 空白页/仅分隔符页丢弃
            continue
        pages.append((page_no, text))

    if not pages:
        raise RuntimeError("pymupdf4llm 未提取到有效文本（可能为扫描版 PDF）")

    full = "\n\n".join(t for _, t in pages)
    return ParseResult(source=pdf_path, engine="pymupdf4llm", text=full, pages=pages)


def _parse_with_pypdf(pdf_path: str) -> ParseResult:
    """降级解析：pypdf / PyPDF2 / langchain PyPDFLoader 依次尝试。"""
    pages: List[str] = []
    try:
        from pypdf import PdfReader
        pages = [(p.extract_text() or "") for p in PdfReader(pdf_path).pages]
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            pages = [(p.extract_text() or "") for p in PdfReader(pdf_path).pages]
        except Exception:
            from langchain_community.document_loaders import PyPDFLoader
            pages = [d.page_content for d in PyPDFLoader(pdf_path).load()]

    indexed: List[Tuple[int, str]] = []
    for i, t in enumerate(pages):
        t = clean_markdown(t)
        if len(t) < MIN_PAGE_CHARS:
            continue
        indexed.append((i + 1, t))          # pypdf 为 0-based，+1 统一成 1-based
    if not indexed:
        raise RuntimeError("pypdf 亦未能提取到文本（可能为扫描版 PDF）")
    return ParseResult(
        source=pdf_path, engine="pypdf",
        text="\n\n".join(t for _, t in indexed), pages=indexed,
    )


def parse_pdf(pdf_path: str) -> ParseResult:
    """解析单个 PDF：PyMuPDF4LLM 优先，失败自动降级 pypdf。"""
    try:
        return _parse_with_pymupdf4llm(pdf_path)
    except Exception as exc:      # noqa: BLE001
        warn = f"PyMuPDF4LLM 解析失败，已降级 pypdf：{exc}"
        log(f"    [警告] {warn[:300]}")
        append_error_log(f"解析降级 - {os.path.basename(pdf_path)}", str(exc))
        res = _parse_with_pypdf(pdf_path)
        res.warning = warn
        return res


# ---------------------------------------------------------------------------
# 豆包多模态嵌入（逐条调用，带重试）
# ---------------------------------------------------------------------------
class DoubaoMultimodalEmbeddings:
    """兼容 langchain Embeddings 协议；逐条调用以适配多模态接口格式。"""

    def __init__(self, api_key: str, base_url: str, model: str, max_retries: int = 3):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _get_single_embedding(self, text: str) -> List[float]:
        import requests

        payload = {
            "model": self.model,
            "input": [{"type": "text", "text": text[:8000] or " "}],
            "encoding_format": "float",
        }
        last_err: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/embeddings/multimodal",
                    headers=self.headers, json=payload, timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                if isinstance(data, list):
                    return data[0]["embedding"]
                return data["embedding"]
            except Exception as exc:      # noqa: BLE001 - 网络类异常统一重试
                last_err = exc
                time.sleep(1.5 * attempt)
        raise RuntimeError(f"嵌入接口连续 {self.max_retries} 次失败：{last_err}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        total = len(texts)
        out: List[List[float]] = []
        for i, t in enumerate(texts):
            if i % 10 == 0 or i == total - 1:
                log(f"    生成向量：{i + 1}/{total}")
            out.append(self._get_single_embedding(t))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._get_single_embedding(text)


# ---------------------------------------------------------------------------
# 向量库构建
# ---------------------------------------------------------------------------
def list_target_pdfs() -> List[str]:
    """扫描 docs，按文件名排序后仅取前 MAX_PDF_COUNT 个 PDF。"""
    if not os.path.isdir(DOCS_FOLDER):
        raise SystemExit(f"docs 目录不存在：{DOCS_FOLDER}")
    all_pdfs = sorted(f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(".pdf"))
    selected = all_pdfs[:MAX_PDF_COUNT]
    skipped = all_pdfs[MAX_PDF_COUNT:]
    log(f"docs 下共发现 {len(all_pdfs)} 个 PDF，本次处理前 {len(selected)} 个")
    for f in selected:
        log(f"    [处理] {f}")
    for f in skipped:
        log(f"    [跳过] {f}")
    return [os.path.join(DOCS_FOLDER, f) for f in selected]


def build_chunks(pdf_paths: List[str]):
    """按页解析并切分，chunk 携带真实页码，返回 (Documents, 统计信息)。"""
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", " ", ""],
    )
    docs, stats = [], []
    for idx, path in enumerate(pdf_paths, start=1):
        name = os.path.basename(path)
        log(f"({idx}/{len(pdf_paths)}) 解析 {name}")
        t0 = time.time()
        res = parse_pdf(path)          # 串行解析，避免内存峰值过高
        elapsed = time.time() - t0

        file_chunks = 0
        for page_no, page_text in res.pages:
            for part in splitter.split_text(page_text):
                docs.append(Document(
                    page_content=part,
                    metadata={
                        "source": res.source,
                        "file_name": name,
                        "parser": res.engine,
                        "page": page_no,          # 已是 1-based，直接可用
                        "chunk_index": file_chunks,
                    },
                ))
                file_chunks += 1
        log(f"    引擎={res.engine} 页数={len(res.pages)} 字符={len(res.text)} "
            f"块数={file_chunks} 耗时={elapsed:.2f}s")
        stats.append({
            "file_name": name, "parser": res.engine, "pages": len(res.pages),
            "chars": len(res.text), "chunks": file_chunks,
            "seconds": round(elapsed, 2), "warning": res.warning,
        })
    return docs, stats


def build_vector_db():
    require_api_key()
    pdf_paths = list_target_pdfs()
    docs, stats = build_chunks(pdf_paths)
    if not docs:
        raise SystemExit("未产生任何文本块，构建中止")
    log(f"文档解析完成，共 {len(docs)} 个文本块，开始写入 Chroma ...")

    from langchain_community.vectorstores import Chroma

    embeddings = DoubaoMultimodalEmbeddings(API_KEY, BASE_URL, EMBEDDING_MODEL)

    # 全新构建：旧库改名保留而非直接删除，可回滚
    if os.path.isdir(CHROMA_DB_PATH):
        backup = CHROMA_DB_PATH + "_old_" + time.strftime("%Y%m%d_%H%M%S")
        log(f"检测到旧向量库，先重命名保留为 {os.path.basename(backup)}")
        os.rename(CHROMA_DB_PATH, backup)

    BATCH = 32
    vector_db = None
    for s in range(0, len(docs), BATCH):
        batch = docs[s:s + BATCH]
        if vector_db is None:
            vector_db = Chroma.from_documents(
                documents=batch, embedding=embeddings,
                persist_directory=CHROMA_DB_PATH, collection_name="ai_standards_v2",
            )
        else:
            vector_db.add_documents(batch)
        log(f"    已写入 {min(s + BATCH, len(docs))}/{len(docs)}")

    if hasattr(vector_db, "persist"):
        try:
            vector_db.persist()
        except Exception:
            pass

    size_mb = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _d, fs in os.walk(CHROMA_DB_PATH) for f in fs
    ) / 1024 / 1024
    summary = {
        "files": stats,
        "total_chunks": len(docs),
        "chroma_path": CHROMA_DB_PATH,
        "chroma_size_mb": round(size_mb, 2),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"向量数据库构建完成：{len(docs)} 块，{size_mb:.2f} MB -> {CHROMA_DB_PATH}")
    log(f"构建摘要：{SUMMARY_PATH}")
    return vector_db, summary


def load_vector_db():
    require_api_key()
    from langchain_community.vectorstores import Chroma

    if not os.path.isdir(CHROMA_DB_PATH):
        raise SystemExit(f"向量库不存在：{CHROMA_DB_PATH}，请先执行 python rag_system_v2.py build")
    embeddings = DoubaoMultimodalEmbeddings(API_KEY, BASE_URL, EMBEDDING_MODEL)
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings, collection_name="ai_standards_v2",
    )


# ---------------------------------------------------------------------------
# 问答逻辑
# ---------------------------------------------------------------------------
def get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=LLM_MODEL, api_key=require_api_key(), base_url=BASE_URL, temperature=0.1,
    )


def answer_question(user_question: str) -> str:
    if not user_question or not user_question.strip():
        return "请输入有效的问题。"

    log(f"收到问题：{user_question}")
    vector_db = load_vector_db()
    retriever = vector_db.as_retriever(search_kwargs={"k": TOP_K})
    docs = retriever.invoke(user_question)
    log(f"检索到 {len(docs)} 段相关文档")

    context_text = "\n\n".join(
        f"【{(d.metadata or {}).get('file_name', '未知')} 第{(d.metadata or {}).get('page', '?')}页】\n{d.page_content}"
        for d in docs
    )
    prompt = f"""请严格根据下方的上下文内容回答用户问题。如果上下文没有相关信息，请回答“暂无对应规范资料”，不要编造。回答要严谨、准确，符合正式文档的表述风格，涉及具体条款时指出出处。

上下文内容：
{context_text}

用户问题：
{user_question}"""

    answer = get_llm().invoke(prompt).content

    sources = []
    for d in docs:
        meta = d.metadata or {}
        name = meta.get("file_name") or os.path.basename(str(meta.get("source", "未知文件")))
        sources.append(f"- 来源：《{name}》 第{meta.get('page', '?')}页")
    unique_sources = list(dict.fromkeys(sources))

    return f"{answer}\n\n---\n**引用依据：**\n" + ("\n".join(unique_sources) if unique_sources else "- 无")


# ---------------------------------------------------------------------------
# Gradio 界面
# ---------------------------------------------------------------------------
def gradio_interface():
    import gradio as gr

    with gr.Blocks(title="人工智能国家标准智能问答系统 V2") as demo:
        gr.Markdown("# 📚 人工智能国家标准智能问答系统 V2")
        gr.Markdown("PyMuPDF4LLM 结构化解析 + RAG 检索增强生成，回答附带原文页码依据")
        question_input = gr.Textbox(
            label="请输入你的问题",
            placeholder="例如：具身智能数据生成平台分为哪几层？",
            lines=2,
        )
        submit_btn = gr.Button("提交提问", variant="primary")
        answer_output = gr.Markdown(label="回答结果")
        submit_btn.click(fn=answer_question, inputs=question_input, outputs=answer_output)
    return demo


# ---------------------------------------------------------------------------
# 验证与状态输出
# ---------------------------------------------------------------------------
def run_verification(question: str) -> str:
    """单次问答验证，结果写入 verify_report.md 并返回报告内容。"""
    started = time.time()
    error_text, answer = "", ""
    try:
        answer = answer_question(question)
    except SystemExit as exc:
        error_text = str(exc)
    except Exception as exc:      # noqa: BLE001
        error_text = f"{type(exc).__name__}: {exc}"
        append_error_log("验证问答失败", error_text)
    elapsed = time.time() - started

    has_citation = "引用依据" in answer and "- 来源：" in answer
    report = "\n".join([
        "# V2 问答验证报告",
        "",
        f"- 验证时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 测试问题：{question}",
        f"- 耗时：{elapsed:.1f} 秒",
        f"- 含引用依据：{'是' if has_citation else '否'}",
        f"- 运行报错：{'无' if not error_text else error_text}",
        "",
        "## 模型回答",
        "",
        answer or "(无回答)",
        "",
    ])
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    log(f"验证报告已写入：{REPORT_PATH}")
    print("\n" + report)
    return report


def show_info() -> None:
    log("===== 环境状态摘要 =====")
    log(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    log(f"VOLC_API_KEY: {'已设置' if os.environ.get('VOLC_API_KEY') else '未设置'}")
    log(f"BASE_URL: {BASE_URL}")
    log(f"LLM_MODEL: {LLM_MODEL}")
    log(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    for mod in ("pymupdf4llm", "pymupdf", "chromadb", "gradio",
                "langchain_text_splitters", "langchain_openai", "langchain_community", "pypdf"):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "")
            log(f"  [OK]   {mod} {ver}")
        except Exception as exc:      # noqa: BLE001
            log(f"  [FAIL] {mod}: {type(exc).__name__}")
    log(f"docs 目录: {DOCS_FOLDER} （存在={os.path.isdir(DOCS_FOLDER)}）")
    log(f"向量库目录: {CHROMA_DB_PATH} （存在={os.path.isdir(CHROMA_DB_PATH)}）")
    if os.path.exists(SUMMARY_PATH):
        with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
            log("上次构建摘要: " + json.dumps(json.load(f), ensure_ascii=False))


def main() -> None:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "build").lower()
    try:
        if mode == "build":
            build_vector_db()
        elif mode == "serve":
            log("系统启动完成")
            gradio_interface().launch(inbrowser=True, share=False)
        elif mode == "test":
            q = sys.argv[2] if len(sys.argv) > 2 else "具身智能数据生成平台分为哪几层？"
            run_verification(q)
        elif mode == "info":
            show_info()
        else:
            raise SystemExit(f"未知模式：{mode}（可用：build / serve / test / info）")
    except SystemExit:
        raise
    except Exception as exc:      # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        append_error_log(mode, detail)
        log(f"[错误] {detail}，详情已写入 {ERROR_LOG_PATH}")
        raise


if __name__ == "__main__":
    main()
