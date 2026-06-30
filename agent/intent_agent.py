import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from openai import OpenAI


class IntentAgent:
    def __init__(self, llm_func):
        # 把普通函数包装成Runnable，管道只会传入字符串
        self.llm = RunnableLambda(llm_func)

    def chat(self, user_query):
        router_prompt_template = PromptTemplate(
            template='''
            你是一个招投标采购智能问答系统中的意图识别专家。请分析用户的问题，判断其属于以下哪一类：
            1."招标信息获取"：用户需要获取准确的"招标信息"，包括招标公告、中标结果、项目信息、公司信息、供应商信息等，需要查询相关数据；
            2."招标政策法律法规咨询"：用户需要询问"与招标、投标、采购业务相关的各种问题"；
            3."无关问题"：与招投标无关的闲聊、天气等。

            规则：
            类别1 返回 {{"1": "bid_info_search"}}
            类别2 返回 {{"2": "law_consult"}}
            类别3 返回 {{"3": "other"}}

            示例：
            示例1：招标金额大于3亿的项目有哪些？:{{"1": "bid_info_search"}}
            示例2：招标人与中标人何时签合同？:{{"2": "law_consult"}}
            示例3：明天天气如何？:{{"3": "other"}}

            只输出JSON，不要多余文字。
            用户问题：{user_query}
            ''',
            input_variables=["user_query"]
        )

        chain = router_prompt_template | self.llm | StrOutputParser()
        response = chain.invoke({"user_query": user_query})
        return response
        
# if __name__ == '__main__':
#     print("=== 招投标智能问答系统 ===")
#     print("输入 'q' 或 'quit' 退出系统")
#     print("输入 'clear' 清除对话历史")
#     print("=" * 40)


#     from models.LLM import get_router_deepseek
#     import json

#     intent_agent = IntentAgent(llm_func=get_router_deepseek)

#     while True:
#         query = input("\n请输入与招投标相关的问题：").strip()
#         if query.lower() in ['q', 'quit', '退出']:
#             print("感谢使用，再见！")
#             break
#         if query.lower() in ['clear', '清空']:
#             print("对话已清空")
#             continue
#         if not query:
#             print("请输入有效问题")
#             continue

#         print("\n正在查询中...")
#         intent_response = intent_agent.chat(query)
#         print(intent_response)

#         #意图识别 并路由到对应的agent
#         try:
#             if intent_response:
#                 if isinstance(intent_response, str):
#                     intent_response = json.loads(intent_response)
#                 intent = next(iter(intent_response.keys()))
#                 intent_name = intent_response[intent]   
#                 print(intent, intent_name) 
#                 # 路由到对应的agent
#                 if intent_name == "bid_info_search":
#                     bidagent = BidAgent(vectorstore, docs_retriever=vector_retrieval.get_bid_docs, llm_api=get_completion_deepseek, docs_nums=10)
#                     bid_response = bidagent.chat(query)
#                     print(bid_response)
#                 elif intent_name == "law_consult":
#                     lawagent = LawAgent(vectorstore, docs_retriever=vector_retrieval.get_law_docs, llm_api=get_completion_deepseek, docs_nums=10)
#                     law_response = lawagent.chat(query)
#                     print(law_response)
#                 elif intent_name == "other":
#                     otheragent = OtherAgent(vectorstore, docs_retriever=vector_retrieval.get_other_docs, llm_api=get_completion_deepseek, docs_nums=10)
#                     other_response = otheragent.chat(query)
#                     print(other_response)
#                 else:
#                     print("无法识别用户意图")
#         except json.JSONDecodeError:
#             print("无法识别用户意图")
#             continue
