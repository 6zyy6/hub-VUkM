import os

os.environ["OPENAI_API_KEY"] = "  "
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from agents import Agent, Runner, function_tool, handoff
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


sentiment_agent = Agent(
    model="qwen-max",
    name="情感分类专家",
    instructions="""你是一个专业的情感分类专家。
请对用户提供的文本进行情感分析，严格按照以下要求：
1. 情感类别只能是：正面、负面、中性 三者之一
2. 给出分类结果后，说明判断的主要依据
3. 给出0-1之间的置信度分数

输出格式示例：
【情感分类结果】：正面
【判断依据】：文本中使用了"开心"、"太棒了"等积极词汇
【置信度】：0.95

只关注情感分类任务，不要回答其他问题。如果用户的请求不是情感分析，请告诉主agent需要转接。
"""
)


entity_agent = Agent(
    model="qwen-max",
    name="实体识别专家",
    instructions="""你是一个专业的实体识别专家。
请对用户提供的文本进行实体识别，严格按照以下要求：
1. 识别并提取以下类型的实体：
   - 人名(PERSON)
   - 地点(LOCATION)
   - 组织(ORGANIZATION)
   - 时间(TIME)
   - 产品(PRODUCT)
2. 每个实体需要标注：实体文本、实体类型、在文本中的位置

输出格式示例：
【实体识别结果】：
1. 张三 - 人名(PERSON) - 位置: 第0-2字
2. 北京 - 地点(LOCATION) - 位置: 第5-7字

只关注实体识别任务，不要回答其他问题。如果用户的请求不是实体识别，请告诉主agent需要转接。
"""
)


main_agent = Agent(
    model="qwen-max",
    name="任务调度Agent",
    instructions="""你是一个任务调度专家。
请分析用户的请求，选择最合适的专家Agent来处理：

【可选择的专家】：
1. 情感分类专家 - 专门处理文本情感分析、情绪判断的任务
2. 实体识别专家 - 专门处理文本中的实体提取、命名实体识别任务

【选择规则】：
- 如果用户需要分析文本的情感、情绪、态度，请转交给"情感分类专家"
- 如果用户需要识别文本中的人名、地名、组织名等实体，请转交给"实体识别专家"
- 如果用户同时需要两种分析，请先转交给"情感分类专家"
- 如果不是以上两类任务，请告知用户支持的任务类型

请使用handoff功能自动转接给对应的专家agent。
""",
    handoffs=[
        handoff(sentiment_agent, 
                tool_name="transfer_to_sentiment_agent",
                tool_description="当用户需要进行情感分析或情绪判断时，调用此工具转接给情感分类专家"),
        handoff(entity_agent,
                tool_name="transfer_to_entity_agent",
                tool_description="当用户需要进行实体识别或命名实体提取时，调用此工具转接给实体识别专家")
    ]
)


print("=" * 60)
print("多Agent NLP分析系统")
print("=" * 60)
print("支持的功能：")
print("  1. 情感分类（正面/负面/中性）")
print("  2. 实体识别（人名/地点/组织/时间/产品）")
print("=" * 60)
print()

test_cases = [
    "帮我分析这句话的情感：今天天气真好，我心情特别开心！",
    "识别这句话中的实体：张三明天要去北京阿里巴巴公司开会",
    "这句话是什么情绪：这家餐厅的服务太差了，饭菜也很难吃",
]

for i, user_input in enumerate(test_cases, 1):
    print(f"\n{'='*60}")
    print(f"测试用例 {i}: {user_input}")
    print(f"{'='*60}")
    
    result = Runner.run_sync(main_agent, user_input)
    print(f"\n最终回答:")
    print(result.final_output)
    print(f"\n处理该请求的Agent: {result.last_agent.name}")
    print("-" * 60)

print("\n\n✅ 所有测试用例执行完成！")
print("\n💡 您也可以自定义输入进行测试：")
print("   result = Runner.run_sync(main_agent, '您的文本')")
print("   print(result.final_output)")
