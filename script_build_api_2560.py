# script_build_api_2560.py
import os
import json
import time
import re
import argparse
import asyncio
import xxhash
import numpy as np
import pdfplumber
import torch
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from graphr1 import GraphR1
from graphr1.utils import wrap_embedding_func_with_attrs


# ---------------------------------------------
# 1. 配置路径与 API
# ---------------------------------------------
# LLM 配置 (DeepSeek API)
API_KEY = "sk-45a3b1bbcdc34df2a9805b7614ac951f" 
BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

# Embedding 配置 (本地 Qwen 模型)
EMBED_MODEL_PATH = "/root/Qwen3-Embedding-4B"

# 数据目录 (保持您脚本中的路径)
DATA_DIR = "/root/Graph-R1/data_for_hypergraph"

# ---------------------------------------------
# 2. 初始化模型
# ---------------------------------------------

# A. 初始化 DeepSeek API 客户端
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
print(f"LLM 客户端已就绪: {MODEL_NAME}")

# B. 初始化本地 Qwen Embedding 模型
print(f"正在加载 Embedding 模型: {EMBED_MODEL_PATH} ...")
try:
    # 尝试开启 flash_attention_2 以加速 (如果显卡支持)
    embed_model = SentenceTransformer(
        EMBED_MODEL_PATH,
        trust_remote_code=True,
        model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"}
    )
except Exception as e:
    print(f"Flash Attention 加载失败，回退到默认模式: {e}")
    embed_model = SentenceTransformer(
        EMBED_MODEL_PATH, 
        trust_remote_code=True,
        device="cuda:1" # 修改为CPU ID 为 1 的 GPU
    )

# 动态获取模型维度 (确保维度参数绝对正确)
EMBEDDING_DIM = embed_model.get_sentence_embedding_dimension()
print(f"Embedding 模型加载完毕，维度: {EMBEDDING_DIM}")

# ---------------------------------------------
# 3. 核心功能函数
# ---------------------------------------------

# --- 新增: 自定义 Embedding 函数 (适配 GraphR1 接口) ---
@wrap_embedding_func_with_attrs(embedding_dim=EMBEDDING_DIM, max_token_size=8192)
async def my_qwen_embedding(texts: list[str], **kwargs) -> np.ndarray:
    """
    使用 Qwen-Embedding-4B 生成向量。
    """
    # 使用 asyncio.to_thread 将同步的 GPU 计算放入线程池，防止阻塞事件循环
    embeddings = await asyncio.to_thread(
        embed_model.encode, 
        texts, 
        convert_to_numpy=True, 
        show_progress_bar=False,
        batch_size=16 # 根据显存情况调整
    )
    return embeddings

# --- 原有: API 调用包装器 ---
async def my_api_llm_call(prompt: str, system_prompt: str = None, history_messages: list = [], **kwargs) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=False,
            temperature=0.0,
            max_tokens=4096
        )
        content = response.choices[0].message.content
        return content if content else ""
    except Exception as e:
        print(f"API 调用失败: {e}")
        return ""

# --- 原有: 高质量 PDF 解析器 ---
def parse_pdf_high_quality(file_path):
    full_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                # 1. 尝试提取表格
                tables = page.extract_tables()
                table_texts = []
                for table in tables:
                    cleaned_table = [[cell if cell else "" for cell in row] for row in table]
                    if cleaned_table:
                        header = " | ".join(cleaned_table[0])
                        separator = " | ".join(["---"] * len(cleaned_table[0]))
                        body = "\n".join([" | ".join(row) for row in cleaned_table[1:]])
                        table_texts.append(f"\n{header}\n{separator}\n{body}\n")

                # 2. 提取正文
                text = page.extract_text(x_tolerance=2, y_tolerance=3)
                
                if text: full_text.append(text)
                if table_texts: full_text.extend(table_texts)
                    
    except Exception as e:
        print(f"PDF 解析失败 {file_path}: {e}")
        return ""
    return "\n".join(full_text)

# --- 新增：读取markdown文件内容的函数 ---
# 读取 data_for_hypergraph/NCCN/文献文件夹/ → 指南名 = "NCCN"，
# 使用正则表达式或读取第一个#标记内容作为标题
# 读取同文件夹下的PMID.txt作为PMID
def load_paper_from_folder(paper_dir: str) -> dict:
    """
    从单个文献文件夹中读取 Paper 元数据：
    - guideline（指南名）
    - title（Markdown 第一个 # 标题）
    - pmid（PMID.txt）
    - content（full.md 全文）
    - /root/Graph-R1/data_for_hypergraph/NCCN/paper_001
    -/root/Graph-R1/data_for_hypergraph/FIGO/paper_001
    """

    # ---------- 1. 基本路径校验 ----------
    if not os.path.isdir(paper_dir):
        raise ValueError(f"不是有效的文献文件夹: {paper_dir}")

    md_path = os.path.join(paper_dir, "full.md")
    pmid_path = os.path.join(paper_dir, "PMID.txt")

    if not os.path.exists(md_path):
        raise FileNotFoundError(f"未找到 full.md: {md_path}")

    # ---------- 2. guideline = 上一级目录名 ----------
    # data_for_hypergraph/NCCN/xxx_paper_folder/
    guideline = os.path.basename(os.path.dirname(paper_dir))

    # ---------- 3. 读取 Markdown 全文 ----------
    with open(md_path, "r", encoding="utf-8") as f:
        markdown_text = f.read().strip()

    # ---------- 4. 使用正则提取标题（第一个 # 开头的行） ----------
    title = None
    for line in markdown_text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            # 去掉所有前导 # 和空格
            title = re.sub(r"^#+\s*", "", line)
            break

    if not title:
        raise ValueError(f"未在 Markdown 中找到标题 (#): {md_path}")

    # ---------- 5. 读取 PMID ----------
    # 如果 PMID.txt 存在且非空，则读取其内容
    # 如果PMID不存在或为空，则用标题的Hash值代替
    pmid = None
    if os.path.isfile(pmid_path):
        with open(pmid_path, 'r', encoding='utf-8') as f:
            pmid = f.read().strip() or None
    if pmid and not pmid.isdigit():  # 非法 PMID 回退
        pmid = None
    if not pmid:  # 确定性哈希
        pmid = str(xxhash.xxh64(title.encode('utf-8')).intdigest())


    # ---------- 6. 返回统一结构 ----------
    return {
        "guideline": guideline,     # 指南
        "title": title,             # 标题
        "pmid": pmid,               # PMID
        "content": markdown_text,   # 全文内容
    }

# ---------------------------------------------
# 4. 主构建逻辑
# ---------------------------------------------
async def extract_knowledge(rag, paper_content, paper_name):
    """
       针对【单篇 Paper】进行知识抽取
       """
    print(f"开始插入文档{paper_name}的相关节点")

    max_retries = 5 # 最大重试次数
    for attempt in range(1, max_retries + 1):
        try:
            await rag.ainsert(
                paper_content,
                paper_name=paper_name
            )
            print(f"✅ 文献 {paper_name} 抽取完成")
            return

        except Exception as e:
            print(f"⚠️ 文献 {paper_name} 第 {attempt}/{max_retries} 次失败: {e}")
            if attempt == max_retries:
                raise
            await asyncio.sleep(5) # 等待后重试

    # batch_size = 50
    # total_batches = (len(unique_contexts) + batch_size - 1) // batch_size
    #
    # for i in range(0, len(unique_contexts), batch_size):
    #     batch_contexts = unique_contexts[i:i + batch_size]
    #     print(f"--- 正在处理批次 {(i // batch_size) + 1}/{total_batches} ---")
    #
    #     retries = 0
    #     while retries < 5:
    #         try:
    #             await rag.ainsert(batch_contexts)
    #             print(f"批次 {(i // batch_size) + 1} 成功插入。")
    #             break
    #         except Exception as e:
    #             retries += 1
    #             print(f"重试 {retries}/5: {e}")
    #             await asyncio.sleep(5)

async def insert_knowledge(rag, paper_content, paper_name):

    # 变化：rag 初始化 转移到了 main 函数中，因为要根据PMID去重，需要访问 rag 实例查询
    # rag = GraphR1(
    #     working_dir=f"expr/{data_source}",
    #
    #     # LLM 部分
    #     llm_model_func=my_api_llm_call,
    #     llm_model_name=MODEL_NAME,
    #
    #     # --- 新增: Embedding 部分 ---
    #     embedding_func=my_qwen_embedding,
    #
    #     # --- 新增: 维度同步 ---
    #     # 必须确保图嵌入(Node2Vec)的维度与文本嵌入维度一致
    #     node2vec_params={
    #         "dimensions": EMBEDDING_DIM,
    #         "num_walks": 10,
    #         "walk_length": 40,
    #         "window_size": 2,
    #         "iterations": 3,
    #         "random_seed": 3,
    #     },
    #
    #     # 其他配置
    #     chunk_token_size=1600,
    #     chunk_overlap_token_size=50,
    #     graph_storage="Neo4JStorage"
    # )
    await extract_knowledge(rag, paper_content, paper_name)
    print(f"知识超图为 '{data_source}' 构建成功。")


# ---------------------------------------------
# 5. 主异步函数
# ---------------------------------------------
async def main():
    # ==================================
    parser = argparse.ArgumentParser()
    # 建议更改 data_source 名称以避免与旧的 1536 维数据冲突
    parser.add_argument("--data_source", type=str, default="DeepSeek_QwenEmbed_Graph")
    args = parser.parse_args()
    # ===================================


    # --- 新增: 初始化 GraphR1 实例 ---
    rag = GraphR1(
        working_dir=f"expr/{args.data_source}",

        # LLM 部分
        llm_model_func=my_api_llm_call,
        llm_model_name=MODEL_NAME,

        # --- 新增: Embedding 部分 ---
        embedding_func=my_qwen_embedding,

        # --- 新增: 维度同步 ---
        # 必须确保图嵌入(Node2Vec)的维度与文本嵌入维度一致
        node2vec_params={
            "dimensions": EMBEDDING_DIM,
            "num_walks": 10,
            "walk_length": 40,
            "window_size": 2,
            "iterations": 3,
            "random_seed": 3,
        },

        # 其他配置
        chunk_token_size=1600,
        chunk_overlap_token_size=50,
        graph_storage="Neo4JStorage"
    )
    print("✅ GraphR1初始化完成")

    # 统计变量
    stats = {
        "total": 0,
        "new": 0,
        "existing": 0,
        "errors": 0
    }


    print(f"开始从 {DATA_DIR} 加载数据...")

    unique_contexts = []

    if not os.path.exists(DATA_DIR):
        print(f"错误：数据目录不存在: {DATA_DIR}")
        exit(1)

    # 遍历目录结构: data_for_hypergraph/{GuidelineName}/{PaperFolder}/full.md
    for guideline_name in os.listdir(DATA_DIR):
        guideline_path = os.path.join(DATA_DIR, guideline_name)
        if not os.path.isdir(guideline_path):
            continue

        print(f"\n📁 处理指南: {guideline_name}")

        for paper_folder in os.listdir(guideline_path):
            paper_path = os.path.join(guideline_path, paper_folder)
            if not os.path.isdir(paper_path):
                continue

            stats["total"] += 1 # 统计总文献数

            try:
                # 1. 获取 paper 元数据
                paper = load_paper_from_folder(paper_path)  # 返回 dict 得到 paper 元数据

                paper_pmid = paper["pmid"]  # 获取pmid
                paper_title = paper["title"]  # 获取标题
                paper_guideline = paper["guideline"]  # 获取指南名
                paper_content = paper["content"]  # 获取全文内容

                # 2. 去重检查：优先通过 PMID 查找库里是否已经有这个 Paper 节点
                existing_name = await rag.chunk_entity_relation_graph.get_paper_by_pmid(paper_pmid) if paper_pmid else None

                # 如果存在，则更新其所属指南列表并跳过 LLM 提取
                if existing_name:
                    print(f"文献 {paper_pmid} 已存在，更新所属指南: {paper_guideline}")
                    await rag.chunk_entity_relation_graph.update_paper_guidelines(existing_name, paper_guideline) # 更新指南列表
                    stats["existing"] += 1
                    continue  # 跳过 LLM 提取，处理下一篇

                # 3. 如果不存在，为新文献，则新建文献节点paper
                stats["new"] += 1
                paper_node_id = f"paper::{paper_pmid}" # 唯一节点名
                print(f"  正在加载新文献: {paper['title']} | PMID: {paper['pmid']} | 所属指南:{paper['guideline']} ")

                # 创建 paper 节点
                await rag.chunk_entity_relation_graph.upsert_node(
                    node_name=paper_node_id,
                    node_data=
                        {
                        "role": "paper",                    # 节点角色为paper
                        "pmid": paper_pmid,
                        "guidelines": [paper_guideline],
                        "title": paper_title,
                        }
                )

                # 4. 读取全文内容，插入知识库
                content = paper_content
                paper_name = f"paper::{paper_pmid}" # 统一的 paper 节点名

                # 传入 paper_name，让后续生成的超边自动关联到它，使用之前的接口，但是新增一个paper_name参数和rag实例
                # 逻辑由多个文件的unique_contexts变成单个文件的content，因为要追溯到paper节点
                try:
                    await insert_knowledge(rag, content, paper_name)
                except Exception as e:
                    print(f"  插入知识库失败 {paper_path} : {e}")
                    stats["errors"] += 1

            except Exception as e:
                print(f"  加载失败 {paper_path} : {e}")

    print("\n--- 文献处理统计 ---")
    print(f"总文献数: {stats['total']}")
    print(f"新文献数: {stats['new']}")
    print(f"已存在文献数: {stats['existing']}")
    print(f"处理错误数: {stats['errors']}")


    # if not unique_contexts:
    #     print("错误：未找到任何有效 Markdown 文献。")
    #     exit(1)
    # print(f"成功加载了 {len(unique_contexts)} 篇文献。")



    # ---- 原有的文件遍历逻辑 (已注释) ----
    # for filename in os.listdir(DATA_DIR):
    #     file_path = os.path.join(DATA_DIR, filename)
    #     try:
    #         if filename.endswith(".txt"):
    #             print(f"  正在加载 (TXT): {filename}")
    #             with open(file_path, 'r', encoding='utf-8') as f:
    #                 unique_contexts.append(f.read())
    #         elif filename.endswith(".jsonl"):
    #             print(f"  正在加载 (JSONL): {filename}")
    #             with open(file_path, 'r', encoding='utf-8') as f:
    #                 for line in f:
    #                     data = json.loads(line)
    #                     if "contents" in data: unique_contexts.append(data["contents"])
    #                     elif "text" in data: unique_contexts.append(data["text"])
    #         elif filename.endswith(".pdf"):
    #             print(f"  正在加载 (PDF): {filename}")
    #             content = parse_pdf_high_quality(file_path)
    #             if len(content) > 50:
    #                 unique_contexts.append(content)
    #             else:
    #                 print(f"  警告: PDF {filename} 内容过短")
    #         else:
    #             print(f"  跳过: {filename}")
    #     except Exception as e:
    #         print(f"读取文件 {filename} 出错: {e}")
    # if not unique_contexts:
    #     print("错误：未找到有效文档。")
    #     exit(1)
    # print(f"成功加载了 {len(unique_contexts)} 个文档。")

    # try:
    #     asyncio.run(insert_knowledge(args.data_source, unique_contexts))
    # except Exception as e:
    #     print(f"构建过程中发生致命错误: {e}")
# ---------------------------------------------
# 6. 程序入口
# ---------------------------------------------
if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())


