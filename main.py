


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("=== 招投标智能问答系统 ===")
    print("输入 'q' 或 'quit' 退出系统")
    print("输入 'clear' 清除对话历史")
    print("=" * 40)

    from models.LLM import get_intent_deepseek
    from agent.intent_agent import IntentAgent
    from agent.bid_agent import BidAgent
    # from agent.law_agent import LawAgent
    # from agent.other_agent import OtherAgent
    # from models.VectorStore import VectorStore
    import json

    intent_agent = IntentAgent(llm_func=get_intent_deepseek)

    while True:
        query = input("\n请输入与招投标相关的问题：").strip()
        if query.lower() in ['q', 'quit', '退出']:
            print("感谢使用，再见！")
            break
        if query.lower() in ['clear', '清空']:
            print("对话已清空")
            continue
        if not query:
            print("请输入有效问题")
            continue

        print("\n正在查询中...")
        intent_response = intent_agent.chat(query)
        print(intent_response)

        #意图识别 并路由到对应的agent
        try:
            if intent_response:
                if isinstance(intent_response, str):
                    intent_response = json.loads(intent_response)
                intent = next(iter(intent_response.keys()))
                intent_name = intent_response[intent]   
                print(intent, intent_name) 
                # 路由到对应的agent
                if intent_name == "bid_info_search":
                    print("招投标信息查询。。。")
                    from retrieval import vector_persist_db,vector_retrieval
                    from models.LLM import get_completion_deepseek
                    vectorstore = vector_persist_db.init_update_bid_data_vectorstore()
                    bidagent = BidAgent(vectorstore, docs_retriever=vector_retrieval.get_bid_docs, llm_api=get_completion_deepseek, docs_nums=10)
                    bid_response = bidagent.chat(query)
                    print(bid_response)
                elif intent_name == "law_consult":
                    print("招标政策法律法规咨询。。。")
                    pass
                    # lawagent = LawAgent(vectorstore, docs_retriever=vector_retrieval.get_law_docs, llm_api=get_completion_deepseek, docs_nums=10)
                    # law_response = lawagent.chat(query)
                    # print(law_response)
                elif intent_name == "other":
                    print("无关问题咨询。。。")
                    pass
                    # otheragent = OtherAgent(vectorstore, docs_retriever=vector_retrieval.get_other_docs, llm_api=get_completion_deepseek, docs_nums=10)
                    # other_response = otheragent.chat(query)
                    # print(other_response)
                else:
                    print("无法识别用户意图")
        except json.JSONDecodeError:
            print("无法识别用户意图")
            continue
