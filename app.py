#!/usr/bin/env python3
import os
import json
# import requests
import httpx
import base64
from dotenv import load_dotenv
from nicegui import ui
from nicegui.events import UploadEventArguments
import asyncio

load_dotenv()


os.environ["DATABRICKS_HOST"] = "https://adb-984752964297111.11.azuredatabricks.net"
os.environ["DATABRICKS_TOKEN"] = os.environ.get("DBSQL_TOKEN")
endpoint_name = "llava-next-mistral7b-sg"
spin = False

# def score_model(payload:dict)-> requests.Response:
#     """
#     Use the deployed model to score and fetch responses.

#     Args:
#         payload (dict): The input payload to be sent to the model.

#     Returns:
#         requests.Response: The response object containing the result of the scoring.

#     Raises:
#         requests.exceptions.RequestException: If an error occurs while making the request.
#     """
#     url = f'{os.environ.get("DATABRICKS_HOST")}/serving-endpoints/{endpoint_name}/invocations'
#     headers = {
#         "Authorization": f'Bearer {os.environ.get("DBSQL_TOKEN")}',
#         "Content-Type": "application/json",
#     }
#     json_payload = json.dumps(payload)
#     response = requests.request(
#         method="POST", headers=headers, url=url, data=json_payload, timeout=90.0
#     )
#     print(response.text)
#     return response

async def ainvoke_model(payload:dict):
    url = f'{os.environ.get("DATABRICKS_HOST")}/serving-endpoints/{endpoint_name}/invocations'
    headers = {
        "Authorization": f'Bearer {os.environ.get("DBSQL_TOKEN")}',
        "Content-Type": "application/json",
    }
    json_payload = json.dumps(payload)
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, headers=headers, data=json_payload)
    return response.json()["predictions"]["candidates"][0].split("\nASSISTANT:")[1]


async def send_to_bricks(e: UploadEventArguments)-> None:
    """
    Sends the uploaded image and user prompt to the bricks for processing.

    Args:
        e (UploadEventArguments): The event arguments containing the uploaded content.

    Returns:
        None
    """
    enc_image = base64.b64encode(e.content.read()).decode("ascii")
    prompt = usr_prompt.value
    print(f"INFO:User prompt value:{prompt}")
    payload = {
        "dataframe_split": {
            "columns": ["prompt", "image"],
            "data": [
                [
                    f"<image>\nUSER:{prompt}\nASSISTANT:",
                    enc_image,
                ]
            ],
        }
    }
    pre_load.text = "I am processing your request. Please wait a moment..."
    response =  await asyncio.gather(ainvoke_model(payload))
    response = response[0].strip()
    print(response)
    # trucnate the response until the last period
    response = response[: response.rfind(".")+1]
    upload.content = response
    pre_load.text = "I have processed your request. Please see the results below."


def reset_text():
    upload.content = ""
    print(f"INFO:Upload text has been cleared. New value:{upload.text}")
    usr_prompt.value = ""
    print(f"INFO:User prompt value has been cleared. New value:{usr_prompt.value}")

with ui.column().style("gap:5em;"):
    ui.colors(primary="#81C783", secondary="green",accent="blue")
    with ui.row():
        dark = ui.dark_mode().style("color: green;")
        ui.button("", on_click=dark.enable, icon="dark_mode")
        ui.button("", on_click=dark.disable, icon="light_mode")
    with ui.column():
        cat_says = ui.chat_message(
            "Hello! I am DetectoCat, a friendly AI detective cat powered by Databricks. I am here to help you with your questions and to provide you with information about the images you upload!",
            name="DetectoCat",
            avatar="https://robohash.org/BWZ.png?set=set4",
        ).style("width: 60em;color:#80780e;")
        usr_prompt = ui.textarea("Enter your Prompt Here").style("width: 40em")
        
        img = (
            ui.upload(on_upload=send_to_bricks, auto_upload=True, label="Click + to Upload Image")
            .style(add="width: 40em; height: 60em; color: #4BB051")
        )
        # ui.spinner('dots', size='lg').bind_visibility(target_object=pre_load, target_name='text',value="I am processing your request. Please wait a moment.")
        pre_load = ui.label("Upload Image above to Interact with Detectocat").style("width: 60em; color: orange;")
        upload = ui.markdown().style("width: 60em; color: green; font-size: 1.25em")
        ui.button("Reset", on_click=reset_text,color='blue').style("width: 15em")
     

ui.run()