
from retrieval.law_vector_persist_db import init_update_law_data_vectorstore
from embeddings.embedding_fn import get_law_embeddings_qwen

class LawAgent:
    law_vector_db = None

    def __init__(self, llm_api, n_result=2):
        if LawAgent.law_vector_db is None:
            LawAgent.law_vector_db = init_update_law_data_vectorstore(
                embedding_fn=get_law_embeddings_qwen,
                collection_name="law_consult"
            )
        self.llm_api = llm_api
        self.n_result = n_result


    def law_chat(self, user_uqery):
        # 1.检索
        search_results = self.law_vector_db.search(user_uqery, self.n_result)
        search_results = "\n".join(search_results["documents"][0])

        # 构建prompt
        law_prompt_template = f'''
         你是一个专业的招投标法律政策咨询专家，你的任务是根据下述给定的已知信息回答用户的问题。
         确保你的回复完全根据下述已知信息，不要编造答案。
         如果下述已知信息不足以回答用户的问题，请直接回复'我无法回答你的问题'。
         已知信息：
         {search_results}
         用户的问题：{user_uqery}

         请用中文回答。
        '''
        print(law_prompt_template)

        # 调用LLM
        response = self.llm_api(law_prompt_template)

        return response



if __name__ == '__main__':
    # 初始化招投标智能助手
    print("=== 招投标智能问答系统 ===")
    print("输入 'q' 或 'quit' 退出系统")
    print("输入 'clear' 清除对话历史")
    print("=" * 40)

    from models.LLM import get_completion_deepseek
    lawagent = LawAgent(llm_api=get_completion_deepseek)
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
        response = lawagent.law_chat(query)
        print(f"\n回答：{response}")
