from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Define the prompt template
template = """
Extract the order id from the following message and format it as a JSON object and use underscore to separate words: {message}
"""

# Initialize the prompt template
prompt = PromptTemplate(input_variables=["message"], template=template)

# Initialize OpenAI LLM
llm = OpenAI()

# Example message
message = "Please cancel order 9"

# Generate the response using LangChain
response = llm(prompt.format(message=message))

# Output the result
print(response)
