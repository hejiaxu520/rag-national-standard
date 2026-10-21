# 1. 导入依赖
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
import gradio as gr
import os
import requests
from typing import List

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FOLDER = os.path.join(BASE_DIR, "docs")
CHROMA_DB_PATH = os.path.join(BASE_DIR, "chroma_db")

# 2. API配置
API_KEY = os.environ.get("VOLC_API_KEY", "")
BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
LLM_MODEL = "doubao-seed-2-0-mini-260428"
EMBEDDING_MODEL = "ep-20261018185559-ptlmc"

# 3. 初始化云端生成模型
llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=API_KEY,
    base_url=BASE_URL,
    temperature=0.1
)

# ========== 修正后的多模态嵌入类：逐个调用，兼容接口格式 ==========
class DoubaoMultimodalEmbeddings(Embeddings):
    def __init__(self, api_key, base_url, model):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _get_single_embedding(self, text: str) -> List[float]:
        """调用单条文本嵌入，兼容多模态接口格式"""
        input_data = [{"type": "text", "text": text}]
        resp = requests.post(
            f"{self.base_url}/embeddings/multimodal",
            headers=self.headers,
            json={
                "model": self.model,
                "input": input_data,
                "encoding_format": "float"
            }
        )
        resp.raise_for_status()
        data = resp.json()
        # 兼容官方返回结构
        if isinstance(data["data"], list):
            return data["data"][0]["embedding"]
        else:
            return data["data"]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档逐个调用，避免批量接口不兼容"""
        embeddings = []
        for i, text in enumerate(texts):
            if i % 5 == 0:
                print(f"正在生成向量：{i+1}/{len(texts)}")
            embeddings.append(self._get_single_embedding(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._get_single_embedding(text)

# 初始化嵌入模型
embeddings = DoubaoMultimodalEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=EMBEDDING_MODEL
)

# 4. 文档分块器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", "。", " ", ""]
)

def build_vector_db():
    all_docs = []
    for file in os.listdir(DOCS_FOLDER):
        if file.lower().endswith(".pdf"):
            file_path = os.path.join(DOCS_FOLDER, file)
            print(f"正在加载文档：{file}")
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            chunks = text_splitter.split_documents(pages)
            all_docs.extend(chunks)
    print(f"文档加载完成，共切分为 {len(all_docs)} 个文本块")
    
    vector_db = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    vector_db.persist()
    print("向量数据库构建完成")
    return vector_db

# 加载或构建向量库
if os.path.exists(CHROMA_DB_PATH):
    vector_db = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )
    print("已加载本地向量数据库")
else:
    vector_db = build_vector_db()

retriever = vector_db.as_retriever(search_kwargs={"k": 4})

# 5. RAG问答逻辑
def answer_question(user_question):
    if not user_question.strip():
        return "请输入有效的问题。"
    
    print(f"收到问题：{user_question}")
    docs = retriever.invoke(user_question)
    print(f"检索到 {len(docs)} 段相关文档")
    
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""请严格根据下方的上下文内容回答用户问题。如果上下文没有相关信息，请回答“暂无对应规范资料”。回答要严谨、准确，符合正式文档的表述风格。

上下文内容：
{context_text}

用户问题：
{user_question}"""
    
    response = llm.invoke(prompt)
    answer = response.content
    
    sources = []
    for doc in docs:
        source_name = os.path.basename(doc.metadata["source"])
        page_num = doc.metadata["page"] + 1
        sources.append(f"- 来源：《{source_name}》 第{page_num}页")
    unique_sources = list(dict.fromkeys(sources))
    
    final_answer = f"{answer}\n\n---\n**引用依据：**\n" + "\n".join(unique_sources)
    return final_answer

# 6. 构建界面
def gradio_interface():
    with gr.Blocks(title="人工智能国家标准智能问答系统") as demo:
        gr.Markdown("# 📚 人工智能国家标准智能问答系统")
        gr.Markdown("基于RAG检索增强生成技术，精准定位标准条款，回答附带原文依据")
        
        question_input = gr.Textbox(
            label="请输入你的问题",
            placeholder="例如：具身智能数据生成平台分为哪几层？",
            lines=2
        )
        submit_btn = gr.Button("提交提问", variant="primary")
        answer_output = gr.Markdown(label="回答结果")
        
        submit_btn.click(
            fn=answer_question,
            inputs=question_input,
            outputs=answer_output
        )
    return demo

if __name__ == "__main__":
    demo = gradio_interface()
    print("系统启动完成，浏览器已自动打开")
    demo.launch(inbrowser=True, share=False)
