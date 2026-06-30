
class BidAgent:
    def __init__(self, vectorstore, docs_retriever, llm_api, docs_nums=10):
        self.vectorstore = vectorstore
        self.docs_retriever = docs_retriever
        self.llm_api = llm_api
        self.docs_nums = docs_nums

    def chat(self, user_query):
        # 1.检索
        # 招投标数据
        search_results = self.docs_retriever(user_query, self.vectorstore, self.docs_nums)

        # search_results = "\n".join(search_results["documents"][0])
        # 构建prompt
        prompt_template = f'''
         你是一个专业的招投标专家，你的任务是根据下述给定的已知信息回答用户的问题。
         确保你的回复完全根据下述已知信息，不要编造答案。
         如果下述已知信息不足以回答用户的问题，请直接回复'我无法回答你的问题'。
         已知招投信息：
         {search_results}
         用户的问题：{user_query}

         请用中文回答。
        '''
        print(prompt_template)

        prompt = prompt_template

        # 调用LLM

        response = self.llm_api(prompt)

        return response


if __name__ == '__main__':

    # 初始化招投标智能助手
    print("=== 招投标智能问答系统 ===")
    print("输入 'q' 或 'quit' 退出系统")
    print("输入 'clear' 清除对话历史")
    print("=" * 40)

    csv_file_path = r"E:\WorkSpace\RAG\bid_datas"


    from retrieval import vector_persist_db,vector_retrieval
    from models.LLM import get_completion_deepseek


    vectorstore = vector_persist_db.init_update_bid_data_vectorstore()
    bidagent = BidAgent(vectorstore, docs_retriever=vector_retrieval.get_bid_docs, llm_api=get_completion_deepseek, docs_nums=10)

    # 对话循环
    while True:
        query = input("\n请输入与招投标相关的问题：").strip()

        # 退出判断
        if query.lower() in ['q', 'quit', '退出']:
            print("感谢使用，再见！")
            break

        # 清除历史（可选功能）
        if query.lower() in ['clear', '清空']:
            print("对话已清空")
            continue

        # 空输入跳过
        if not query:
            print("请输入有效问题")
            continue

        # 执行查询并输出结果
        print("\n正在查询中...")
        response = bidagent.chat(query)
        print(f"\n回答：{response}")
