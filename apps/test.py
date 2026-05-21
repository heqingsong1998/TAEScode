from openai import OpenAI

# 1. 填入你从中转站获取的 API Key
API_KEY = "sk-814bd2b4622fc98e57c348efff908f478d15a1e456217ef8b13c367a1b1ce70a"

# 2. 中转站的接口地址（代码已配置好）
BASE_URL = "https://api.zcsj.top/v1"

def test_connection():
    # 初始化客户端，替换默认的 OpenAI 地址
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL
    )

    print(f"正在尝试连接接口: {BASE_URL} ...\n")

    try:
        # 发送一个简单的问候请求
        response = client.chat.completions.create(
            model="gpt-5.5",  # 如果中转站支持 gpt-4，这里也可以换成 gpt-4
            messages=[
                {"role": "user", "content": "你好，这是一个API连通性测试。如果你收到了，请只回复“连接成功”四个字。"}
            ],
            max_tokens=20
        )
        
        print("✅ 连通测试通过！")
        print("AI 返回内容：", response.choices[0].message.content)

    except Exception as e:
        print("❌ 请求失败！错误信息如下：\n")
        print(e)

if __name__ == "__main__":
    test_connection()