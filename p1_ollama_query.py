from ollama import chat
from ollama import ChatResponse

ai_model = 'gpt-oss:120b-cloud'
demo_prompt = 'define types of numbers in 1 line? integer, whole, natural'

response: ChatResponse = chat(model=ai_model, messages=
[
  {
    'role': 'user',
    'content': demo_prompt,
  },
])

print(response.message.content)
