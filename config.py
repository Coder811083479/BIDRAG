# config.py
import os
from dotenv import load_dotenv

LOAD_ENV_STATUS = load_dotenv()

# 项目根目录（以当前 config.py 文件所在目录为基准）
# __file__ = 当前文件的绝对路径
# os.path.dirname(__file__) = config.py 所在的目录（即 demo_with_langgraph/）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_abs_path(relative_path: str) -> str:
    """
    将相对路径转换为绝对路径，基准为 BASE_DIR（DAG/）
    无论从哪个目录运行程序，解析结果都是正确的
    """
    return os.path.normpath(os.path.join(BASE_DIR, relative_path))

class Config:

    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
    DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://api.dashscope.com")
    
    FINAL_ANSWER_LLM_MODEL_NAME = "deepseek-v4-pro"
    INTENTION_LLM_MODEL_NAME = ROUTER_LLM_MODEL_NAME = "deepseek-v4-flash"
    BGE_M3_MODEL_PATH = r"D:/hugging_face/modelscope/bge-m3"

    # Chroma 向量库路径（推荐用这个，自动转为绝对路径）
    # 相对路径基准：demo_with_langgraph/ 目录
    CHROMA_PERSIST_DIR = get_abs_path(r"./chroma_db/")

    # 下面两个保留，方便你对比
    # CHROMA_PERSIST_ABSOLUTE_DIR = r""
    # CHROMA_PERSIST_RELATIVE_DIR = r"./chroma_db/legal_articles/bge-embedding/all_laws_add_documents"

    # 向量配置（BGE-M3 维度）
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 1024))  # BGE-M3 默认输出维度是1024 #

if __name__ == '__main__':
    pass
    print(Config.DEEPSEEK_API_KEY)
    # print(os.getenv('DASHSCOPE_API_KEY'))