# Databricks notebook source
# %pip install jsonformer
%pip install --upgrade transformers==4.39.0
%pip install optimum
%pip install accelerate>=0.22.0
# %pip install --upgrade auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
# %pip install autoawq
%pip install rich

dbutils.library.restartPython()

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %md
# MAGIC If you are resource strapped --> You could pick up a quantized model and run with it!

# COMMAND ----------

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name_or_path = "TheBloke/llava-v1.5-13B-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-32g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
prompt = "Tell me about AI"
prompt_template=f'''{prompt}
'''
print("\n\n*** Generate:")
input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

# print("*** Pipeline:")
# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=512,
#     do_sample=True,
#     temperature=0.7,
#     top_p=0.95,
#     top_k=40,
#     repetition_penalty=1.1
# )

# print(pipe(prompt_template)[0]['generated_text'])

# COMMAND ----------

prompt = "Tell me about Databricks"
prompt_template=f'''{prompt}
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Non-quantized versions natively supported in transformers

# COMMAND ----------

# del(model)
# import torch
# torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### How about we do this with IMAGES! 🌄

# COMMAND ----------

# MAGIC %md
# MAGIC #### Usecase 1: Explain the Image

# COMMAND ----------

from PIL import Image
import requests
from rich import print
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf", device_map="auto")


# model = LlavaForConditionalGeneration.from_pretrained("liuhaotian/llava-v1.6-34b", device_map="auto")
# processor = AutoProcessor.from_pretrained("liuhaotian/llava-v1.6-34b", device_map="auto")

prompt = "<image>\nUSER: Explain the content of the image?\nASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device="cuda")
# Generate
generate_ids = model.generate(**inputs, max_length=300)
processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# COMMAND ----------

# MAGIC %md
# MAGIC Ofcourse, the model can see - its multi-modal, but how is it useful to me? Answer: **Personalize** the prompt. Your *data* is your secret weapon!

# COMMAND ----------

# MAGIC %md
# MAGIC #### Usecase 2: Personalized multi-modal Q&A Bot
# MAGIC

# COMMAND ----------

prompt = "<image>\nUSER: I really like this style, specifically the jeans. What can this be paired with? I'm someone who prefers wearing sweatshirts. Will I like this style or would you recommend some other jeans? Here's some additional context about the user. This user is very `quality conscious` - doesn't mind to spend some additional dollars and time researching the right product. With this info, give them in depth info about the product's quality and value using the following jeans as options - Rovis Jeans Slim fit  \nASSISTANT:"

url = "https://richmedia.ca-richimage.com/ImageDelivery/imageService?profileId=12026540&id=1859027&recipeId=728"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

# Generate
generate_ids = model.generate(**inputs, max_length=500)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
result = output[0]
print(result.split("\nASSISTANT:")[1].strip())

# COMMAND ----------

# MAGIC %md
# MAGIC BUT, I dont care about yapping away like a bot. I use it on my **backend** pipelines...I need just the JSON

# COMMAND ----------

# MAGIC %md
# MAGIC #### Usecase 3: JSON-Forming for Data Pipelines
# MAGIC

# COMMAND ----------

prompt = "<image>\nUSER: Explain the clothing, specifically the jeans? What is it? The color and the sex of the person wearing it? and the material? Only return JSON for the above items and nothing else\nASSISTANT:"

url = "https://richmedia.ca-richimage.com/ImageDelivery/imageService?profileId=12026540&id=1859027&recipeId=728"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

# Generate
generate_ids = model.generate(**inputs, max_length=100)
output = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
result = output[0]
print(result.split("\nASSISTANT:")[1].strip())

# COMMAND ----------

# MAGIC %md
# MAGIC Ok, can I then use it for product quality issues?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Usecase 4: Augmented Anomaly Detection

# COMMAND ----------

prompt = "<image>\nUSER: Explain the problem in this picture with the product. A customer complained about this product\nASSISTANT:"

url = "https://m.media-amazon.com/images/I/51a94AxNRPL.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

# Generate
generate_ids = model.generate(**inputs, max_length=200)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
result = output[0]
print(result.split("\nASSISTANT:")[1].strip())

# COMMAND ----------

# MAGIC %md
# MAGIC Push it harder - maybe can we use it at the store?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Usecase 5: Vision Augmented Stock Out Prediction

# COMMAND ----------

prompt = "<image>\nUSER: Is there an out of stock situation? What type of product seems to be out of stock? (food/clothing/appliances/durables)\nASSISTANT:"

url = "https://assets.eposnow.com/public/content-images/pexels-roy-broo-empty-shelves-grocery-items.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")

# Generate
generate_ids = model.generate(**inputs, max_length=200)
output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
result = output[0]
print(result.split("\nASSISTANT:")[1].strip())

# COMMAND ----------


import torch
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC # What next? Depending on the usecase
# MAGIC 1. You can package the model via MLflow 📦📦
# MAGIC 2. Serve it up on an endpoint 🏋️‍♀️🏋️‍♀️
# MAGIC 3. Use for downstream apps 👨‍💻👨‍💻
# MAGIC 4. OR Use it in your backend spark jobs! ✨✨

# COMMAND ----------

# MAGIC %md
# MAGIC But can we go one step further? Let's package the model into Model Registry!

# COMMAND ----------

from huggingface_hub import snapshot_download
snapshot_location = snapshot_download(repo_id="llava-hf/llava-1.5-7b-hf")

# COMMAND ----------


import mlflow.pyfunc
import mlflow
import requests
import pandas as pd
import base64
import io
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import base64

class Llava7b15(mlflow.pyfunc.PythonModel):
  
    def load_context(self, context):
        """Method to initialize the model and tokenizer."""

        self.model = LlavaForConditionalGeneration.from_pretrained(
          context.artifacts['repository'],device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(context.artifacts['repository'], device_map="auto")
        self.model.eval()
    
    def _generate_response(self, usr_prompt, image):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        # Send to model and generate a response
        inputs = self.processor(text=usr_prompt, images=image, return_tensors="pt").to(device="cuda")

        output = self.model.generate(
            **inputs,
            max_new_tokens=150,
        )
        result = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return result

    def predict(self, context, model_input):
        """Method to generate predictions for the given input."""
        outputs = []
        for i in range(len(model_input)):
            usr_prompt = model_input["prompt"][i]
            image = model_input["image"][i]
            decoded_image = base64.b64decode(image)
            image = Image.open(io.BytesIO(decoded_image))
            generated_data = self._generate_response(usr_prompt,image)
            outputs.append(generated_data)
        return {"candidates": outputs}

# Define input and output schema for the model
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

input_schema = Schema([ColSpec(DataType.string, "prompt"), ColSpec(DataType.string, "image")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
response = requests.get(url)
image = response.content
encoded_image = base64.b64encode(image).decode("ascii")
input_example = pd.DataFrame({
    "prompt":["<image>\nUSER: What's the content of the image?\nASSISTANT:"],
    "image":[encoded_image]
})

# COMMAND ----------

# prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
# url = "https://www.ilankelman.org/stopsigns/australia.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=prompt, images=image, return_tensors="pt").to(device="cuda")
# # Generate
# generate_ids = model.generate(**inputs, max_length=30)
# processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


# COMMAND ----------

# Log the model using MLflow
import os
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"]="false"
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=Llava7b15(),
        artifacts={'repository' : snapshot_location},
        input_example=input_example,
        pip_requirements=["torch==2.0.1","transformers==4.38.2", "cloudpickle==2.0.0","accelerate>=0.28.0","torchvision==0.15.2","optimum==1.17.1"],
        signature=signature
    )

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")
MODEL_NAME = "main.sgfs.llava7bv15"
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    MODEL_NAME,
)

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
client = MlflowClient()
mlflow.set_registry_uri("databricks-uc")
# # Annotate the model as "CHAMPION".
MODEL_NAME = "main.sgfs.llava7bv15"
client.set_registered_model_alias(name=MODEL_NAME, alias="Champion", version=result.version)
# Load it back from UC
import mlflow
loaded_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Champion")

# COMMAND ----------

import base64
url = "https://richmedia.ca-richimage.com/ImageDelivery/imageService?profileId=12026540&id=1859027&recipeId=728"
response = requests.get(url)
image = response.content
encoded_image = base64.b64encode(image).decode("ascii")

# COMMAND ----------

loaded_model.predict(
    {
        "prompt": ["<image>\nUSER: What's the content of the image?\nASSISTANT:"],
        "image":encoded_image,
    }
)

# COMMAND ----------


# from PIL import Image
# import io

# # Decode the base64 encoded image
# decoded_image = base64.b64decode(encoded_image)

# # Load the decoded image as a raw image
# image = Image.open(io.BytesIO(decoded_image))
# image

# COMMAND ----------

endpoint_name = 'llava-7b-v15-sg'

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

import requests
import json

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

model_version = result  # the returned result of mlflow.register_model
served_name = f'{model_version.name.replace(".", "_")}_{model_version.version}'

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
workload_type = "GPU_LARGE"

endpoint_config = {
  "name": endpoint_name,
  "config": {
    "served_models": [{
      "name": served_name,
      "model_name": model_version.name,
      "model_version": model_version.version,
      "workload_type": workload_type,
      "workload_size": "Small",
      "scale_to_zero_enabled": "False"
    }]
  }
}
endpoint_json = json.dumps(endpoint_config, indent='  ')
# Send a POST request to the API
deploy_response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)
if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')

# COMMAND ----------

print(deploy_response.json())

# COMMAND ----------

# MAGIC %md #### Expand into Llava-Next
# MAGIC
# MAGIC

# COMMAND ----------

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests
from rich import print
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
#loading in half precision format
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", 
                                                          torch_dtype=torch.float16, 
                                                          low_cpu_mem_usage=True,
                                                          use_flash_attention_2=True) 
model.to("cuda:0")

# COMMAND ----------

# prepare image and text prompt, using the appropriate prompt template
url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "[INST] <image>\nWhat is shown in this image? Keep it brief[/INST]"
# below is prompt format for vicuna
# prompt = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat is shown in this image? Explain the content ASSISTANT:"""
inputs = processor(prompt, image, return_tensors="pt").to("cuda:0")
# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=200, pad_token_id=processor.tokenizer.pad_token_id, eos_token_id = processor.tokenizer.eos_token_id)
result = processor.decode(output[0], skip_special_tokens=True)

# COMMAND ----------

# Import matplotlib library
import matplotlib.pyplot as plt

# Display the image using matplotlib 
plt.imshow(image)
plt.axis('off')
plt.show()

# COMMAND ----------

print(result.split("[/INST]")[1].strip())

# COMMAND ----------

# MAGIC %md Log Model to MLFlow!
# MAGIC

# COMMAND ----------

from huggingface_hub import snapshot_download
snapshot_location = snapshot_download(repo_id="llava-hf/llava-v1.6-mistral-7b-hf")

# COMMAND ----------


import mlflow.pyfunc
import mlflow
import requests
import pandas as pd
import base64
import io
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import base64

class LlavaNext7bMistral(mlflow.pyfunc.PythonModel):
  
    def load_context(self, context):
        """Method to initialize the model and processor."""

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
          context.artifacts['repository'],device_map="auto",
          torch_dtype=torch.float16, 
          low_cpu_mem_usage=True,
        )
        self.processor = LlavaNextProcessor.from_pretrained(context.artifacts['repository'], device_map="auto")
        self.model.eval()
    
    def _generate_response(self, usr_prompt, image):
        """
        This method generates prediction for a single input.
        """
        # Build the prompt
        # Send to model and generate a response
        inputs = self.processor(text=usr_prompt, images=image, return_tensors="pt").to(device="cuda")

        output = self.model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=processor.tokenizer.pad_token_id, 
            eos_token_id=processor.tokenizer.eos_token_id
        )
        result = self.processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return result

    def predict(self, context, model_input):
        """Method to generate predictions for the given input."""
        outputs = []
        for i in range(len(model_input)):
            usr_prompt = model_input["prompt"][i]
            image = model_input["image"][i]
            # decode and read the image
            decoded_image = base64.b64decode(image)
            image = Image.open(io.BytesIO(decoded_image))
            generated_data = self._generate_response(usr_prompt,image)
            outputs.append(generated_data)
        return {"candidates": outputs}

# Define input and output schema for the model
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec

input_schema = Schema([ColSpec(DataType.string, "prompt"), ColSpec(DataType.string, "image")])
output_schema = Schema([ColSpec(DataType.string)])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
response = requests.get(url)
image = response.content
encoded_image = base64.b64encode(image).decode("ascii")
input_example = pd.DataFrame({
    "prompt":["<image>\nUSER: What's the content of the image?\nASSISTANT:"],
    "image":[encoded_image]
})

# COMMAND ----------

# Log the model using MLflow
import os
os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"]="false"
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=LlavaNext7bMistral(),
        artifacts={'repository' : snapshot_location},
        input_example=input_example,
        pip_requirements=["torch==2.0.1","transformers==4.39.0", "cloudpickle==2.0.0","accelerate==0.25.0","torchvision==0.15.2","optimum==1.18.0"],
        signature=signature
    )

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")
MODEL_NAME = "main.sgfs.llava-next-7b-mistral"
result = mlflow.register_model(
    "runs:/"+run.info.run_id+"/model",
    MODEL_NAME,
)

# COMMAND ----------

import mlflow
from mlflow import MlflowClient
client = MlflowClient()
mlflow.set_registry_uri("databricks-uc")
# # Annotate the model as "CHAMPION".
client.set_registered_model_alias(name=MODEL_NAME, alias="Champion", version=result.version)
# Load it back from UC
import mlflow
loaded_model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@Champion")

# COMMAND ----------

loaded_model.predict(
    {
        "prompt": ["[INST] <image>\nWhat is shown in this image? Keep it brief[/INST]"],
        "image":encoded_image,
    }
)

# COMMAND ----------

endpoint_name = 'llava-next-mistral7b-sg'

databricks_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

import requests
import json

deploy_headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
deploy_url = f'{databricks_url}/api/2.0/serving-endpoints'

model_version = result  # the returned result of mlflow.register_model
served_name = f'{model_version.name.replace(".", "_")}_{model_version.version}'

# Specify the type of compute (CPU, GPU_SMALL, GPU_MEDIUM, etc.)
workload_type = "GPU_LARGE"

endpoint_config = {
  "name": endpoint_name,
  "config": {
    "served_models": [{
      "name": served_name,
      "model_name": model_version.name,
      "model_version": model_version.version,
      "workload_type": workload_type,
      "workload_size": "Small",
      "scale_to_zero_enabled": "False"
    }]
  }
}
endpoint_json = json.dumps(endpoint_config, indent='  ')
# Send a POST request to the API
deploy_response = requests.request(method='POST', headers=deploy_headers, url=deploy_url, data=endpoint_json)
if deploy_response.status_code != 200:
  raise Exception(f'Request failed with status {deploy_response.status_code}, {deploy_response.text}')
