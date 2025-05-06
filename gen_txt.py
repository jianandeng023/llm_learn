from transformers import AutoModelForCausalLM, AutoTokenizer
"""
文本生成小项目整体流程
1.加载模型和分词器，注意模型和分词器要匹配。
2.设置prompt，prompt中需要包含角色信息，定义对话历史。
3.将对话历史作为模板应用，并生成格式化文本
4.将格式化文本变成模型可接受的输入格式，
5.调用模型生成响应，并通过max_new_tokens参数控制生成文本的长度。
6.将生成的响应解码成文本并只截取生成的文本部分。
7.输出响应。
"""

model_name = "D:\hug_model\models\Qwen\Qwen2___5-0___5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto" # 使用这个参数需要按照accelerate库
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=128
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(response)
# print("the length of response is ", len(response))