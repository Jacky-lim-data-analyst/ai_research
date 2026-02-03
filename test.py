# from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Instantiate the LLM model
model = ChatOllama(
    model="granite4:3b",
    temperature=0.7,
    base_url="http://192.168.0.162:11434"
)

# class Trivia(BaseModel):
#     question: str = Field(description="The trivia question")
#     answer: str = Field(description="The correct answer to the trivia question")

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    "Tell me a joke about {topic}"
)

# Create a structured LLM using the `with_structured_output` method
# structured_llm = model.with_structured_output(Trivia)

# Chain the prompt and structured LLM using the pipe operator
trivia_chain = prompt | model

# Invoke the chain
result = trivia_chain.invoke({"topic": "space"})

# if hasattr(result, "model_dump"):
#     data = result.model_dump()
# else:
#     data = result.dict()

# print(type(result))
# print(type(data), data)
print(result.content)
