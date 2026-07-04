
class OtherAgent:
    def __init__(self, llm_api):
        self.llm_api = llm_api

    def chat(self, user_query):
        # 构建prompt
        other_prompt_template = f'''
         你是一个智能助手，能友好文明地问答用户提出的问题，如果不知道，就回复用户："这个问题超出了我的能力范围，请重新提问。"

         用户的问题：{user_query}
         
         回复要求符合以下规则：
         1.符合法律法规，文明作答。
         2.回答尽量言简意赅，不要冗余回复。
         3.符合道德标准。
         请用中文回答。
        '''
        print(other_prompt_template)

        # 调用LLM
        response = self.llm_api(other_prompt_template)

        return response


if __name__ == '__main__':

    # 初始化招投标智能助手
    print("=== 招投标智能问答系统 ===")
    print("输入 'q' 或 'quit' 退出系统")
    print("输入 'clear' 清除对话历史")
    print("=" * 40)

    from models.LLM import get_completion_deepseek

    otheragent = OtherAgent(llm_api=get_completion_deepseek)

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
        response = otheragent.chat(query)
        print(f"\n回答：{response}")
