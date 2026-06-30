from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from config import Config

api_key = Config.DEEPSEEK_API_KEY
base_url = Config.DEEPSEEK_BASE_URL

# 初始化llm（通义千问）
llm = ChatTongyi(dashscope_api_key=api_key, model="qwen-max")
embeddings_model = DashScopeEmbeddings(model="text-embedding-v1")


def get_bid_docs(query, vectorstore, docs_nums=10):
    # 字段描述，和你原来保持一致
    field_desc = """

            招投标信息，元数据字段包括：id(项目序号), 发布时间(发布日期), 类别, 来源, 省份, 市, 区县, 项目编号, 采购人(招标方), 代理机构, 预算(万元), 项目地址, 项目周期, 中标人(中标方), 中标金额(元), 中标日期(中标时间)。查询函数使用规则：等于用 eq, 不等于用 ne, 大于用 gt, 大于等于用 gte, 小于用 lt, 小于等于用 lte, 包含用 contain(单数形式，不要用contains), 模糊匹配用 like, 在列表中用 in。请使用精确的字段名和函数名进行查询，不要使用繁体字。

            AttributeInfo(name="id", description="招标项目序号", type="string"),
            AttributeInfo(name="发布时间", description="招标项目发布时间，格式如 2025-01-01，也可称为发布日期", type="string"),
            AttributeInfo(name="类别", description="招标项目类别", type="string"),
            AttributeInfo(name="来源", description="招标项目来源", type="string"),
            AttributeInfo(name="省份", description="招标项目所在省份", type="string"),
            AttributeInfo(name="市", description="招标项目所在市", type="string"),
            AttributeInfo(name="区县", description="招标项目所在区县", type="string"),
            AttributeInfo(name="项目编号", description="招标项目项目编号", type="string"),
            AttributeInfo(name="采购人", description="招标项目采购人，也可称为招标方", type="string"),
            AttributeInfo(name="代理机构", description="招标项目代理机构", type="string"),
            AttributeInfo(name="预算(万元)", description="招标项目预算(万元)", type="float"),
            AttributeInfo(name="项目地址", description="招标项目项目地址", type="string"),
            AttributeInfo(name="项目周期", description="招标项目项目周期", type="string"),
            AttributeInfo(name="中标人", description="招标项目中标人，也可称为中标方", type="string"),
            AttributeInfo(name="中标金额(元)", description="招标项目中标金额(元)", type="float"),
            AttributeInfo(name="中标日期", description="中标日期，格式如 2025-01-01，也可称为中标时间", type="string")

            字段清单（严格使用这些键名）：
            "发布时间"：字符串日期
            "省份"：省份文字
            "预算(万元)"：数字，支持gt、gte、lt、lte
            "类别"：项目分类
            "中标日期"：字符串日期

            语法规则：
            相等 {"省份":"安徽省"}
            大于 {"预算(万元)":{"$gt":500}}
            区间 {"预算(万元)":{"$gte":100,"$lte":2000}}
            多条件同时满足直接合并key。
            没有条件返回空字典 {}。

            只输出纯JSON，不要任何解释文字。
            """

    prompt = PromptTemplate(
        template=field_desc.replace('{', '{{').replace('}', '}}') + "\n用户问题：{q}\n输出JSON：",
        input_variables=["q"]
    )

    # 构造解析链路
    parse_chain = prompt | llm | JsonOutputParser()

    try:
        filter_dict = parse_chain.invoke({"q": query})
    except:
        filter_dict = {}

    # 直接调用向量库带filter检索
    if not filter_dict:
        docs = vectorstore.similarity_search(query, k=docs_nums)
    else:
        docs = vectorstore.similarity_search(query, k=docs_nums, filter=filter_dict)

    return docs