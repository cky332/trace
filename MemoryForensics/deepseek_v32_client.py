"""
独立的 DeepSeek-V3.2 调用客户端（经 SiliconFlow OpenAI 兼容接口）

用法：
    pip install openai
    export SILICONFLOW_API_KEY="sk-xxx"
    python deepseek_v32_client.py

复用：
    from deepseek_v32_client import call_deepseek
    text, usage = call_deepseek("你的 prompt")
"""
import os
import time
from openai import OpenAI

# ============ 配置 ============
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"

# 惰性初始化客户端：仅在真正发起调用时才要求 API key，
# 这样不需要联网的离线流程（如 --reframe_rag）可以正常导入本模块。
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("请设置环境变量 SILICONFLOW_API_KEY")
        _client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _client


# ============ 主调用函数（含重试） ============
def call_deepseek(
    prompt: str,
    system: str = "You are a helpful assistant.",
    temperature: float = 0.5,
    max_tokens: int = 4000,
    top_p: float = 1.0,
    model: str = MODEL,
    max_retries: int = 60,
    return_reasoning: bool = False,
    verbose_retry: bool = True,
):
    """
    调 DeepSeek-V3.2。

    参数：
        prompt: 用户 prompt
        system: 系统提示词
        temperature/max_tokens/top_p: 标准采样参数
        model: 默认 Pro/deepseek-ai/DeepSeek-V3.2，可换其他兼容模型
        max_retries: 限流时最大重试次数
        return_reasoning: True 时额外返回 reasoning_content（思维链）
    返回：
        return_reasoning=False → (content, usage_dict)
        return_reasoning=True  → (content, reasoning_content, usage_dict)
        usage_dict = {"prompt": int, "completion": int, "total": int}
    """
    client = _get_client()
    retries = 0
    last_err = ""
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None) or ""
            usage = {
                "prompt": response.usage.prompt_tokens,
                "completion": response.usage.completion_tokens,
                "total": response.usage.total_tokens,
            }
            return (content, reasoning, usage) if return_reasoning else (content, usage)
        except Exception as e:
            err = str(e)
            last_err = err
            low = err.lower()
            # 限流 / 上游饱和：等更久；其他错误：指数退避
            is_rate_limit = (
                "429" in err or "503" in err
                or any(k in low for k in ["rate", "quota", "overload", "busy", "capacity", "limit"])
                or "饱和" in err or "繁忙" in err
            )
            wait = min(120, 30 + retries * 10) if is_rate_limit else min(60, 2 ** min(retries, 6))
            if verbose_retry:
                print(f"[deepseek] error: {err[:160]}; retry {retries+1}/{max_retries} after {wait}s")
            time.sleep(wait)
            retries += 1
    raise RuntimeError(f"Max retries ({max_retries}) reached. Last error: {last_err}")


# ============ 简单 demo ============
if __name__ == "__main__":
    print("=" * 60)
    print("调用 DeepSeek-V3.2，请稍候 ...\n")

    content, reasoning, usage = call_deepseek(
        prompt="What is 2 + 2? Answer briefly.",
        return_reasoning=True,
    )
    print("【content（最终回答）】")
    print(content)
    print("\n" + "=" * 60)
    print("【reasoning_content（思维链）】")
    print(reasoning if reasoning else "(none)")
    print("\n" + "=" * 60)
    print("【usage】")
    for k, v in usage.items():
        print(f"  {k}: {v}")
    # 粗略成本估算（实际以 SiliconFlow 控制台计费为准）
    cost = (usage["prompt"] * 2 + usage["completion"] * 8) / 1_000_000
    print(f"\n【estimated cost】 ¥{cost:.6f}")
