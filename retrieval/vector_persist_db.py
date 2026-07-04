
import os
from langchain_chroma import Chroma
import pandas as pd
from langchain_core.documents import Document

"""
投招标数据持久化为ChromaDB向量数据库 和增量更新向量数据库
"""
def init_update_bid_data_vectorstore():
    '''
    初始化招投标数据向量数据库,检查是否有新文件或文件已更新，更新向量数据库
    :param csv_file_path: 招投标数据文件夹路径（扫描文件夹下所有CSV）
    :return: vectorstore  返回招投标数据向量数据库
    '''

    from embeddings.embedding_fn import DashScopeEmbedding
    embedding_model = DashScopeEmbedding()

    # 向量数据库持久化路径
    persist_dir = r"E:\WorkSpace\RAG\db\bid_chroma_db"
    os.makedirs(persist_dir, exist_ok=True)

    # 元数据记录文件（记录已处理的文件信息）
    metadata_file = os.path.join(persist_dir, "processed_files.json")

    csv_file_path = r"E:\WorkSpace\RAG\data_sources\bid_datas"
    # 扫描文件夹下所有CSV文件
    data_dir = csv_file_path
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    print(f"扫描到 {len(csv_files)} 个CSV文件: {csv_files}")

    # 计算每个文件的哈希值（文件名 + 修改时间），使用 || 分隔符避免与文件名冲突
    def get_file_hash(filepath):
        mtime = os.path.getmtime(filepath)
        filename = os.path.basename(filepath)
        return f"{filename}||{mtime}"

    current_files_hash = {get_file_hash(os.path.join(data_dir, f)) for f in csv_files}

    print(current_files_hash)

    # 加载已处理的文件记录（支持新格式，兼容处理旧格式）
    processed_files = set()
    file_format_changed = False
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            import json
            old_data = json.load(f)
            old_processed = old_data.get("processed_files", [])
            # 检查是否为新格式（使用 || 分隔符）
            if old_processed and "||" not in str(old_processed[0]):
                print("检测到旧格式的 processed_files.json，需要重新构建向量库...")
                file_format_changed = True
                old_processed = []
            processed_files = set(old_processed)

    # 检查是否有新文件或文件已更新
    new_files = current_files_hash - processed_files
    need_update = len(new_files) > 0

    if need_update:
        print(f"检测到 {len(new_files)} 个新文件/更新文件，需要增量更新向量库...")

        # 加载已有向量库（如果存在）
        if os.path.exists(os.path.join(persist_dir, "chroma.sqlite3")):
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_model,
                collection_name="bid_self_query"
            )
        else:
            # 首次创建空向量库
            vectorstore = Chroma(
                persist_directory=persist_dir,
                embedding_function=embedding_model,
                collection_name="bid_self_query"
            )

        # 处理每个新文件
        for file_hash in new_files:
            filename = file_hash.split("||")[0]
            filepath = os.path.join(data_dir, filename)
            print(f"正在处理文件: {filename}")

            # 读取CSV
            csv_encoding = "gbk"
            df = pd.read_csv(
                filepath,
                encoding=csv_encoding,
                sep=",",
                on_bad_lines="warn"
            )
            print(f"读取 {len(df)} 行数据")

            # 创建Document对象
            docs = []
            for idx, row in df.iterrows():
                # doc = Document(
                #     page_content=row["项目名称"],
                #     metadata={
                #         "id": row["id"],
                #         "发布时间": row["发布时间"],
                #         "类别": row["类别"],
                #         "来源": row["来源"],
                #         "省份": row["省份"],
                #         "市": row["市"],
                #         "区县": row["区县"],
                #         "项目编号": row["项目编号"],
                #         "采购人": row["采购人"],
                #         "代理机构": row["代理机构"],
                #         "预算(万元)": float(row["预算(万元)"]),
                #         "项目地址": row["项目地址"],
                #         "项目周期": row["项目周期"],
                #         "中标人": row["中标人"],
                #         "中标金额(元)": float(row["中标金额(元)"]),
                #         "中标日期": row["中标日期"]
                #     }
                # )
                doc = Document(
                        page_content=f"""
                                    "id": {row["id"]},
                                    "项目名称":{row["项目名称"]},
                                    "发布时间": {row["发布时间"]},
                                    "类别": {row["类别"]},
                                    "来源": {row["来源"]},
                                    "省份": {row["省份"]},
                                    "市": {row["市"]},
                                    "区县": {row["区县"]},
                                    "项目编号": {row["项目编号"]},
                                    "采购人": {row["采购人"]},
                                    "代理机构": {row["代理机构"]},
                                    "预算(万元)": {float(row["预算(万元)"])},
                                    "项目地址": {row["项目地址"]},
                                    "项目周期": {row["项目周期"]},
                                    "中标人": {row["中标人"]},
                                    "中标金额(元)": {float(row["中标金额(元)"])},
                                    "中标日期": {row["中标日期"]}
                                """,
                        metadata={
                            "id":idx
                        },
                        id = idx
                )
                docs.append(doc)

            # 追加到向量库（分批处理，ChromaDB最大批次5461）
            batch_size = 5000
            total_docs = len(docs)
            for i in range(0, total_docs, batch_size):
                batch_docs = docs[i:i + batch_size]
                vectorstore.add_documents(batch_docs)
                print(f"已追加批次 {i // batch_size + 1}/{(total_docs - 1) // batch_size + 1}，{len(batch_docs)} 条文档")
            print(f"共追加 {total_docs} 条文档到向量库")

        # 更新已处理文件记录
        processed_files.update(new_files)
        with open(metadata_file, 'w', encoding='utf-8') as f:
            import json as json_module
            json_module.dump({"processed_files": list(processed_files)}, f)

        print("向量数据库增量更新完成")
    else:
        print("所有文件已处理，直接加载向量数据库...")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model,
            collection_name="bid_self_query"
        )
        print("向量数据库加载完成")

    return vectorstore