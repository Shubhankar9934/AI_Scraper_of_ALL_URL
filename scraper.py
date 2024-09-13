import os

# Disable XNNPACK delegate
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import random
import time
import re
import json
from datetime import datetime
from typing import List, Dict, Type
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, create_model
import html2text
import tiktoken
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
from urllib.parse import urljoin
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


from openai import OpenAI
import google.generativeai as genai
from groq import Groq


from assets import USER_AGENTS,PRICING,HEADLESS_OPTIONS,SYSTEM_MESSAGE,USER_MESSAGE,LLAMA_MODEL_FULLNAME,GROQ_LLAMA_MODEL_FULLNAME
load_dotenv()

# Set up the Chrome WebDriver options

def setup_selenium():
    options = Options()

    # Randomly select a user agent from the imported list
    user_agent = random.choice(USER_AGENTS)
    options.add_argument(f"user-agent={user_agent}")
    options.add_argument("--headless")  # Run Chrome in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    # Add other options
    for option in HEADLESS_OPTIONS:
        options.add_argument(option)

    # Specify the path to the ChromeDriver
    service = Service(r"C:\Users\shubh\driver\chromedriver.exe")  

    # Initialize the WebDriver
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def click_accept_cookies(driver):
    """
    Tries to find and click on a cookie consent button. It looks for several common patterns.
    """
    try:
        # Wait for cookie popup to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//button | //a | //div"))
        )
        
        # Common text variations for cookie buttons
        accept_text_variations = [
            "accept", "agree", "allow", "consent", "continue", "ok", "I agree", "got it"
        ]
        
        # Iterate through different element types and common text variations
        for tag in ["button", "a", "div"]:
            for text in accept_text_variations:
                try:
                    # Create an XPath to find the button by text
                    element = driver.find_element(By.XPATH, f"//{tag}[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text}')]")
                    if element:
                        element.click()
                        print(f"Clicked the '{text}' button.")
                        return
                except:
                    continue

        print("No 'Accept Cookies' button found.")
    
    except Exception as e:
        print(f"Error finding 'Accept Cookies' button: {e}")

def fetch_html_selenium(url):
    driver = setup_selenium()
    try:
        driver.get(url)
        
        # Add random delays to mimic human behavior
        time.sleep(1)  # Adjust this to simulate time for user to read or interact
        driver.maximize_window()
        

        # Try to find and click the 'Accept Cookies' button
        # click_accept_cookies(driver)

        # Add more realistic actions like scrolling
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Simulate time taken to scroll and read
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        html = driver.page_source
        return html
    finally:
        driver.quit()

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove headers and footers based on common HTML tags or classes
    for element in soup.find_all(['header', 'footer']):
        element.decompose()  # Remove these tags and their content

    return str(soup)


def html_to_markdown_with_readability(html_content):

    
    cleaned_html = clean_html(html_content)  
    
    # Convert to markdown
    markdown_converter = html2text.HTML2Text()
    markdown_converter.ignore_links = False
    markdown_content = markdown_converter.handle(cleaned_html)
    
    return markdown_content


    
def save_raw_data(raw_data, timestamp, output_folder='output'):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the raw markdown data with timestamp in filename
    raw_output_path = os.path.join(output_folder, f'rawData_{timestamp}.md')
    with open(raw_output_path, 'w', encoding='utf-8') as f:
        f.write(raw_data)
    print(f"Raw data saved to {raw_output_path}")
    return raw_output_path


def remove_urls_from_file(file_path):
    # Regex pattern to find URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # Construct the new file name
    base, ext = os.path.splitext(file_path)
    new_file_path = f"{base}_cleaned{ext}"

    # Read the original markdown content
    with open(file_path, 'r', encoding='utf-8') as file:
        markdown_content = file.read()

    # Replace all found URLs with an empty string
    cleaned_content = re.sub(url_pattern, '', markdown_content)

    # Write the cleaned content to a new file
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)
    print(f"Cleaned file saved as: {new_file_path}")
    return cleaned_content


def create_dynamic_listing_model(field_names: List[str]) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model based on provided fields.
    field_name is a list of names of the fields to extract from the markdown.
    """
    # Create field definitions using aliases for Field parameters
    field_definitions = {field: (str, ...) for field in field_names}
    # Dynamically create the model with all field
    return create_model('DynamicListingModel', **field_definitions)


def create_listings_container_model(listing_model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Create a container model that holds a list of the given listing model.
    """
    return create_model('DynamicListingsContainer', listings=(List[listing_model], ...))




def trim_to_token_limit(text, model, max_tokens=120000):
    encoder = tiktoken.encoding_for_model(model)
    tokens = encoder.encode(text)
    if len(tokens) > max_tokens:
        trimmed_text = encoder.decode(tokens[:max_tokens])
        return trimmed_text
    return text

def generate_system_message(listing_model: BaseModel) -> str:
    """
    Dynamically generate a system message based on the fields in the provided listing model.
    """
    # Use the model_json_schema() method to introspect the Pydantic model
    schema_info = listing_model.model_json_schema()

    # Extract field descriptions from the schema
    field_descriptions = []
    for field_name, field_info in schema_info["properties"].items():
        # Get the field type from the schema info
        field_type = field_info["type"]
        field_descriptions.append(f'"{field_name}": "{field_type}"')

    # Create the JSON schema structure for the listings
    schema_structure = ",\n".join(field_descriptions)

    # Generate the system message dynamically
    system_message = f"""
    You are an intelligent text extraction and conversion assistant. Your task is to extract structured information 
                        from the given text and convert it into a pure JSON format. The JSON should contain only the structured data extracted from the text, 
                        with no additional commentary, explanations, or extraneous information. 
                        You could encounter cases where you can't find the data of the fields you have to extract or the data will be in a foreign language.
                        Please process the following text and provide the output in pure JSON format with no words before or after the JSON:
    Please ensure the output strictly follows this schema:

    {{
        "listings": [
            {{
                {schema_structure}
            }}
        ]
    }} """

    return system_message



# def format_data(data, DynamicListingsContainer, DynamicListingModel, selected_model):
#     token_counts = {}
    
#     if selected_model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
#         # Use OpenAI API
#         client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
#         completion = client.beta.chat.completions.parse(
#             model=selected_model,
#             messages=[
#                 {"role": "system", "content": SYSTEM_MESSAGE},
#                 {"role": "user", "content": USER_MESSAGE + data},
#             ],
#             response_format=DynamicListingsContainer
#         )
#         # Calculate tokens using tiktoken
#         encoder = tiktoken.encoding_for_model(selected_model)
#         input_token_count = len(encoder.encode(USER_MESSAGE + data))
#         output_token_count = len(encoder.encode(json.dumps(completion.choices[0].message.parsed.dict())))
#         token_counts = {
#             "input_tokens": input_token_count,
#             "output_tokens": output_token_count
#         }
#         return completion.choices[0].message.parsed, token_counts

#     elif selected_model == "gemini-1.5-flash":
#         # Use Google Gemini API
#         genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#         model = genai.GenerativeModel('gemini-1.5-flash',
#                 generation_config={
#                     "response_mime_type": "application/json",
#                     "response_schema": DynamicListingsContainer
#                 })
#         prompt = SYSTEM_MESSAGE + "\n" + USER_MESSAGE + data
#         # Count input tokens using Gemini's method
#         input_tokens = model.count_tokens(prompt)
#         completion = model.generate_content(prompt)
#         # Extract token counts from usage_metadata
#         usage_metadata = completion.usage_metadata
#         token_counts = {
#             "input_tokens": usage_metadata.prompt_token_count,
#             "output_tokens": usage_metadata.candidates_token_count
#         }
#         return completion.text, token_counts
    
#     elif selected_model == "Llama3.1 8B":

#         # Dynamically generate the system message based on the schema
#         sys_message = generate_system_message(DynamicListingModel)
#         # print(SYSTEM_MESSAGE)
#         # Point to the local server
#         client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

#         completion = client.chat.completions.create(
#             model=LLAMA_MODEL_FULLNAME, #change this if needed (use a better model)
#             messages=[
#                 {"role": "system", "content": sys_message},
#                 {"role": "user", "content": USER_MESSAGE + data}
#             ],
#             temperature=0.7,
            
#         )

#         # Extract the content from the response
#         response_content = completion.choices[0].message.content
#         print(response_content)
#         # Convert the content from JSON string to a Python dictionary
#         parsed_response = json.loads(response_content)
        
#         # Extract token usage
#         token_counts = {
#             "input_tokens": completion.usage.prompt_tokens,
#             "output_tokens": completion.usage.completion_tokens
#         }

#         return parsed_response, token_counts
#     elif selected_model== "Groq Llama3.1 70b":
        
#         # Dynamically generate the system message based on the schema
#         sys_message = generate_system_message(DynamicListingModel)
#         # print(SYSTEM_MESSAGE)
#         # Point to the local server
#         client = Groq(api_key=os.environ.get("GROQ_API_KEY"),)

#         completion = client.chat.completions.create(
#         messages=[
#             {"role": "system","content": sys_message},
#             {"role": "user","content": USER_MESSAGE + data}
#         ],
#         model=GROQ_LLAMA_MODEL_FULLNAME,
#     )

#         # Extract the content from the response
#         response_content = completion.choices[0].message.content
        
#         # Convert the content from JSON string to a Python dictionary
#         parsed_response = json.loads(response_content)
        
#         # completion.usage
#         token_counts = {
#             "input_tokens": completion.usage.prompt_tokens,
#             "output_tokens": completion.usage.completion_tokens
#         }

#         return parsed_response, token_counts
#     else:
#         raise ValueError(f"Unsupported model: {selected_model}")



def save_formatted_data(formatted_data, timestamp, output_folder='output'):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Parse the formatted data if it's a JSON string (from Gemini API)
    if isinstance(formatted_data, str):
        try:
            formatted_data_dict = json.loads(formatted_data)
        except json.JSONDecodeError:
            raise ValueError("The provided formatted data is a string but not valid JSON.")
    else:
        # Handle data from OpenAI or other sources
        formatted_data_dict = formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data

    # Save the formatted data as JSON with timestamp in filename
    json_output_path = os.path.join(output_folder, f'sorted_data_{timestamp}.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_data_dict, f, indent=4)
    print(f"Formatted data saved to JSON at {json_output_path}")

    # Prepare data for DataFrame
    if isinstance(formatted_data_dict, dict):
        # If the data is a dictionary containing lists, assume these lists are records
        data_for_df = next(iter(formatted_data_dict.values())) if len(formatted_data_dict) == 1 else formatted_data_dict
    elif isinstance(formatted_data_dict, list):
        data_for_df = formatted_data_dict
    else:
        raise ValueError("Formatted data is neither a dictionary nor a list, cannot convert to DataFrame")

    # Create DataFrame
    try:
        df = pd.DataFrame(data_for_df)
        print("DataFrame created successfully.")

        # Save the DataFrame to an Excel file
        excel_output_path = os.path.join(output_folder, f'sorted_data_{timestamp}.xlsx')
        df.to_excel(excel_output_path, index=False)
        print(f"Formatted data saved to Excel at {excel_output_path}")
        
        return df
    except Exception as e:
        print(f"Error creating DataFrame or saving Excel: {str(e)}")
        return None


from transformers import LlamaTokenizer

def chunk_text(text, model, max_tokens=10000):
    """
    Splits the input text into chunks based on the token limit of the model.
    Uses Hugging Face's LLaMA tokenizer if the model is LLaMA.
    """
    if 'llama' in model.lower():
        # Use a valid Hugging Face model identifier
        tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
        tokens = tokenizer.encode(text)
        # Split tokens into chunks
        chunks = [tokenizer.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    else:
        # Use tiktoken for other models
        try:
            encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base tokenizer if model-specific tokenizer is not available
            print(f"Could not automatically map {model} to a tokenizer. Using 'cl100k_base' as a fallback.")
            encoder = tiktoken.get_encoding("cl100k_base")
        
        tokens = encoder.encode(text)
        # Split tokens into chunks
        chunks = [encoder.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    
    return chunks


# def chunk_text(text, model, max_tokens=10000):
#     """
#     Splits the input text into chunks based on the token limit of the model.
#     Uses Hugging Face's LLaMA tokenizer if the model is LLaMA.
#     """
#     if 'llama' in model.lower():
#         # Load the LLaMA tokenizer from Hugging Face
#         tokenizer = LlamaTokenizer.from_pretrained("huggingface/llama-model",use_auth_token="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")
#         tokens = tokenizer.encode(text)
#         # Split tokens into chunks
#         chunks = [tokenizer.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
#     else:
#         # Use tiktoken for other models
#         try:
#             encoder = tiktoken.encoding_for_model(model)
#         except KeyError:
#             # Fallback to cl100k_base tokenizer if model-specific tokenizer is not available
#             print(f"Could not automatically map {model} to a tokenizer. Using 'cl100k_base' as a fallback.")
#             encoder = tiktoken.get_encoding("cl100k_base")
        
#         tokens = encoder.encode(text)
#         # Split tokens into chunks
#         chunks = [encoder.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    
#     return chunks

# def chunk_text(text, model, max_tokens=10000):
#     """
#     Splits the input text into chunks based on the token limit of the model.
#     Uses an explicit tokenizer if it cannot automatically map the model.
#     """
#     try:
#         encoder = tiktoken.encoding_for_model(model)
#     except KeyError:
#         # Use a fallback encoding (e.g., cl100k_base) if the model-specific tokenizer isn't available
#         print(f"Could not automatically map {model} to a tokenizer. Using 'cl100k_base' as a fallback.")
#         encoder = tiktoken.get_encoding("cl100k_base")
    
#     tokens = encoder.encode(text)
    
#     # Split tokens into chunks that don't exceed max_tokens
#     chunks = [encoder.decode(tokens[i:i + max_tokens]) for i in range(0, len(tokens), max_tokens)]
    
#     return chunks



# def format_data(data, DynamicListingsContainer, DynamicListingModel, selected_model):
#     token_counts = {}
    
#     # Chunk the input data to handle large token limits
#     chunks = chunk_text(data, selected_model)

#     all_responses = []
#     total_input_tokens = 0
#     total_output_tokens = 0

#     for chunk in chunks:
#         if selected_model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
#             # Use OpenAI API
#             client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
#             completion = client.beta.chat.completions.parse(
#                 model=selected_model,
#                 messages=[
#                     {"role": "system", "content": SYSTEM_MESSAGE},
#                     {"role": "user", "content": USER_MESSAGE + chunk},
#                 ],
#                 response_format=DynamicListingsContainer
#             )
#             # Calculate tokens using tiktoken
#             encoder = tiktoken.encoding_for_model(selected_model)
#             input_token_count = len(encoder.encode(USER_MESSAGE + chunk))
#             output_token_count = len(encoder.encode(json.dumps(completion.choices[0].message.parsed.dict())))
#             token_counts = {
#                 "input_tokens": input_token_count,
#                 "output_tokens": output_token_count
#             }
#             total_input_tokens += input_token_count
#             total_output_tokens += output_token_count
#             all_responses.append(completion.choices[0].message.parsed)

#         elif selected_model == "gemini-1.5-flash":
#             # Use Google Gemini API
#             genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#             model = genai.GenerativeModel('gemini-1.5-flash',
#                     generation_config={
#                         "response_mime_type": "application/json",
#                         "response_schema": DynamicListingsContainer
#                     })
#             prompt = SYSTEM_MESSAGE + "\n" + USER_MESSAGE + chunk
#             # Count input tokens using Gemini's method
#             input_tokens = model.count_tokens(prompt)
#             completion = model.generate_content(prompt)
#             # Extract token counts from usage_metadata
#             usage_metadata = completion.usage_metadata
#             token_counts = {
#                 "input_tokens": usage_metadata.prompt_token_count,
#                 "output_tokens": usage_metadata.candidates_token_count
#             }
#             total_input_tokens += input_tokens
#             total_output_tokens += usage_metadata.candidates_token_count
#             all_responses.append(completion.text)

#         elif selected_model == "Llama3.1 8B":
#             # Dynamically generate the system message based on the schema
#             sys_message = generate_system_message(DynamicListingModel)
#             # Point to the local server
#             client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

#             completion = client.chat.completions.create(
#                 model=LLAMA_MODEL_FULLNAME,  # change this if needed
#                 messages=[
#                     {"role": "system", "content": sys_message},
#                     {"role": "user", "content": USER_MESSAGE + chunk}
#                 ],
#                 temperature=0.7,
#             )

#             # Extract the content from the response
#             response_content = completion.choices[0].message.content
#             parsed_response = json.loads(response_content)

#             token_counts = {
#                 "input_tokens": completion.usage.prompt_tokens,
#                 "output_tokens": completion.usage.completion_tokens
#             }
#             total_input_tokens += completion.usage.prompt_tokens
#             total_output_tokens += completion.usage.completion_tokens
#             all_responses.append(parsed_response)

#         elif selected_model == "Groq Llama3.1 70b":
#             # Dynamically generate the system message based on the schema
#             sys_message = generate_system_message(DynamicListingModel)
#             # Point to the local server
#             client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

#             completion = client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": sys_message},
#                     {"role": "user", "content": USER_MESSAGE + chunk}
#                 ],
#                 model=GROQ_LLAMA_MODEL_FULLNAME,
#             )

#             response_content = completion.choices[0].message.content
#             parsed_response = json.loads(response_content)

#             token_counts = {
#                 "input_tokens": completion.usage.prompt_tokens,
#                 "output_tokens": completion.usage.completion_tokens
#             }
#             total_input_tokens += completion.usage.prompt_tokens
#             total_output_tokens += completion.usage.completion_tokens
#             all_responses.append(parsed_response)

#         else:
#             raise ValueError(f"Unsupported model: {selected_model}")

#     # Aggregate all the responses
#     final_response = {
#         "listings": [item for response in all_responses for item in response.get('listings', [])]
#     }

#     # Return the aggregated response and the token counts
#     token_counts = {
#         "input_tokens": total_input_tokens,
#         "output_tokens": total_output_tokens
#     }
#     return final_response, token_counts

def format_data(data, DynamicListingsContainer, DynamicListingModel, selected_model):
    token_counts = {}
    
    # Chunk the input data to handle large token limits
    chunks = chunk_text(data, selected_model)

    all_responses = []
    total_input_tokens = 0
    total_output_tokens = 0

    for chunk in chunks:
        if selected_model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
            # Use OpenAI API
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            completion = client.beta.chat.completions.parse(
                model=selected_model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": USER_MESSAGE + chunk},
                ],
                response_format=DynamicListingsContainer
            )
            # Calculate tokens using tiktoken
            encoder = tiktoken.encoding_for_model(selected_model)
            input_token_count = len(encoder.encode(USER_MESSAGE + chunk))
            output_token_count = len(encoder.encode(json.dumps(completion.choices[0].message.parsed.dict())))
            token_counts = {
                "input_tokens": input_token_count,
                "output_tokens": output_token_count
            }
            total_input_tokens += input_token_count
            total_output_tokens += output_token_count
            all_responses.append(completion.choices[0].message.parsed)

        elif selected_model == "gemini-1.5-flash":
            # Use Google Gemini API
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-1.5-flash',
                    generation_config={
                        "response_mime_type": "application/json",
                        "response_schema": DynamicListingsContainer
                    })
            prompt = SYSTEM_MESSAGE + "\n" + USER_MESSAGE + chunk
            # Count input tokens using Gemini's method
            input_tokens = model.count_tokens(prompt)
            completion = model.generate_content(prompt)
            # Extract token counts from usage_metadata
            usage_metadata = completion.usage_metadata
            token_counts = {
                "input_tokens": usage_metadata.prompt_token_count,
                "output_tokens": usage_metadata.candidates_token_count
            }
            total_input_tokens += input_tokens
            total_output_tokens += usage_metadata.candidates_token_count
            all_responses.append(completion.text)

        elif selected_model == "Llama3.1 8B":
            # Dynamically generate the system message based on the schema
            sys_message = generate_system_message(DynamicListingModel)
            # Point to the local server
            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

            completion = client.chat.completions.create(
                model=LLAMA_MODEL_FULLNAME,  # change this if needed
                messages=[
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": USER_MESSAGE + chunk}
                ],
                temperature=0.7,
            )

            # Print the raw response to inspect it
            response_content = completion.choices[0].message.content
            #print("Raw Model Response:", response_content)  # <-- Add this line to see the raw output
            
            # Convert the content from JSON string to a Python dictionary
            try:
                parsed_response = json.loads(response_content)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue  # Skip this chunk if JSON is malformed
            
            token_counts = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens
            }
            total_input_tokens += completion.usage.prompt_tokens
            total_output_tokens += completion.usage.completion_tokens
            all_responses.append(parsed_response)

        elif selected_model == "Groq Llama3.1 70b":
            # Dynamically generate the system message based on the schema
            sys_message = generate_system_message(DynamicListingModel)
            # Point to the local server
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": USER_MESSAGE + chunk}
                ],
                model=GROQ_LLAMA_MODEL_FULLNAME,
            )

            response_content = completion.choices[0].message.content
            
            # Print the raw response to inspect it
            # print("Raw Model Response:", response_content)  # <-- Add this line to see the raw output

            try:
                parsed_response = json.loads(response_content)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue  # Skip this chunk if JSON is malformed

            token_counts = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens
            }
            total_input_tokens += completion.usage.prompt_tokens
            total_output_tokens += completion.usage.completion_tokens
            all_responses.append(parsed_response)

        else:
            raise ValueError(f"Unsupported model: {selected_model}")

    # Aggregate all the responses
    final_response = {
        "listings": [item for response in all_responses for item in response.get('listings', [])]
    }

    # Return the aggregated response and the token counts
    token_counts = {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens
    }
    return final_response, token_counts


def calculate_price(token_counts, model):
    input_token_count = token_counts.get("input_tokens", 0)
    output_token_count = token_counts.get("output_tokens", 0)
    
    # Calculate the costs
    input_cost = input_token_count * PRICING[model]["input"]
    output_cost = output_token_count * PRICING[model]["output"]
    total_cost = input_cost + output_cost
    
    return input_token_count, output_token_count, total_cost


def fetch_html_with_auto_pagination(url):
    """
    Automatically detect pagination on any website and scrape all pages.
    """
    driver = setup_selenium()
    all_html_pages = []  # List to store HTML content from all pages
    
    try:
        driver.get(url)
        
        # Add a loop to scrape all pages until there is no "Next" button
        while True:
            # Add random delays to mimic human behavior
            time.sleep(1)
            driver.maximize_window()

            # Try to find and click the 'Accept Cookies' button if it appears
            click_accept_cookies(driver)
            
            # Add scrolling actions to simulate user behavior
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Get the current page's HTML and store it
            html = driver.page_source
            all_html_pages.append(html)
            
            # Try to detect the "Next" button or any pagination elements
            try:
                # Check if "Next" or page links exist (common pagination links/buttons)
                next_button = driver.find_element(By.XPATH, "//a[contains(text(), 'Next') or contains(text(), '→') or contains(@aria-label, 'Next')]")
                if next_button.is_enabled():
                    next_button.click()  # Click the "Next" button
                    time.sleep(3)  # Wait for the next page to load
                else:
                    print("No more pages to load.")
                    break
            except:
                print("Pagination ended or 'Next' button not found.")
                break

        return all_html_pages  # Return the HTML content of all pages

    finally:
        driver.quit()

def fetch_and_process_pagination(url, fields):
    """
    Automatically detect pagination, process each page, and save data after scraping each page.
    """
    driver = setup_selenium()
    page_count = 1  # To track which page is being processed
    
    try:
        driver.get(url)
        
        # Loop through pages and process each one individually
        while True:
            print(f"Processing page {page_count}...")
            
            # Add random delays to mimic human behavior
            time.sleep(1)
            driver.maximize_window()

            # Try to find and click the 'Accept Cookies' button if it appears
            click_accept_cookies(driver)
            
            # Add scrolling actions to simulate user behavior
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # Get the current page's HTML
            html = driver.page_source
            
            # Convert HTML to markdown for readability
            markdown = html_to_markdown_with_readability(html)
            
            # Generate timestamp for file names
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Save raw markdown data for this page
            raw_data_path = save_raw_data(markdown, f'{timestamp}_page_{page_count}')
            
            # Create the dynamic listing model for fields
            DynamicListingModel = create_dynamic_listing_model(fields)
            
            # Create the container model that holds a list of the dynamic listing models
            DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
            
            # Format data for this page
            formatted_data, token_counts = format_data(markdown, DynamicListingsContainer, DynamicListingModel, "Groq Llama3.1 70b")
            
            # Save formatted data for this page
            save_formatted_data(formatted_data, f'{timestamp}_page_{page_count}')
            
            # Convert formatted data to text for token counting
            formatted_data_text = json.dumps(formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data)
            
            # Automatically calculate the token usage and cost for the page
            input_tokens, output_tokens, total_cost = calculate_price(token_counts, "Groq Llama3.1 70b")
            print(f"Page {page_count} - Input tokens: {input_tokens}, Output tokens: {output_tokens}, Total cost: ${total_cost:.4f}")
            
            # Try to detect the "Next" button or any pagination elements
            try:
                next_button = driver.find_element(By.XPATH, "//a[contains(text(), 'Next') or contains(text(), '→') or contains(@aria-label, 'Next')]")
                if next_button.is_enabled():
                    next_button.click()  # Click the "Next" button
                    time.sleep(3)  # Wait for the next page to load
                    page_count += 1  # Increment page count
                else:
                    print("No more pages to load.")
                    break
            except:
                print("Pagination ended or 'Next' button not found.")
                break

    finally:
        driver.quit()

# Function to fetch all image URLs from the webpage
def fetch_image_urls(driver, url):
    driver.get(url)
    time.sleep(3)  # Wait for the page to load completely
    
    # Find all <img> elements on the page
    img_elements = driver.find_elements(By.TAG_NAME, 'img')
    
    # Extract the src attributes (image URLs)
    img_urls = []
    for img in img_elements:
        img_url = img.get_attribute('src')
        if img_url:
            # Make sure to convert relative URLs to absolute URLs
            img_url = urljoin(url, img_url)
            img_urls.append(img_url)
    
    return img_urls

# Function to fetch all image URLs from the webpage
def fetch_image_urls(driver, url):
    driver.get(url)
    time.sleep(3)  # Wait for the page to load completely
    
    # Find all <img> elements on the page
    img_elements = driver.find_elements(By.TAG_NAME, 'img')
    
    # Extract the src attributes (image URLs)
    img_urls = []
    for img in img_elements:
        img_url = img.get_attribute('src')
        if img_url:
            # Make sure to convert relative URLs to absolute URLs
            img_url = urljoin(url, img_url)
            img_urls.append(img_url)
    
    return img_urls

# Function to download images from the URLs and save them locally
def download_images(img_urls, output_folder='images'):
    os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
    
    for i, img_url in enumerate(img_urls):
        try:
            # Send a request to the image URL
            response = requests.get(img_url)
            if response.status_code == 200:
                # Save the image file
                img_filename = os.path.join(output_folder, f'image_{i+1}.jpg')
                with open(img_filename, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {img_filename}")
            else:
                print(f"Failed to download {img_url}")
        except Exception as e:
            print(f"Error downloading {img_url}: {e}")

# Main function to fetch and download images
def scrape_images(url):
    driver = setup_selenium()
    try:
        # Fetch image URLs from the page
        img_urls = fetch_image_urls(driver, url)
        
        # Download the images
        download_images(img_urls)
    finally:
        driver.quit()

# if __name__ == "__main__":
#     url = 'https://webscraper.io/test-sites/e-commerce/static'
#     fields=['Name of item', 'Price']

#     try:
#         # # Generate timestamp
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
#         # Scrape data
#         raw_html = fetch_html_selenium(url)
    
#         markdown = html_to_markdown_with_readability(raw_html)
        
#         # Save raw data
#         save_raw_data(markdown, timestamp)

#         # Create the dynamic listing model
#         DynamicListingModel = create_dynamic_listing_model(fields)

#         # Create the container model that holds a list of the dynamic listing models
#         DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
        
#         # Format data
#         formatted_data, token_counts = format_data(markdown, DynamicListingsContainer,DynamicListingModel,"Groq Llama3.1 70b")  # Use markdown, not raw_html
#         print(formatted_data)
#         # Save formatted data
#         save_formatted_data(formatted_data, timestamp)

#         # Convert formatted_data back to text for token counting
#         formatted_data_text = json.dumps(formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data) 
        
        
#         # Automatically calculate the token usage and cost for all input and output
#         input_tokens, output_tokens, total_cost = calculate_price(token_counts, "Groq Llama3.1 70b")
#         print(f"Input token count: {input_tokens}")
#         print(f"Output token count: {output_tokens}")
#         print(f"Estimated total cost: ${total_cost:.4f}")

#     except Exception as e:
#         print(f"An error occurred: {e}")

# Example usage:
if __name__ == "__main__":
    #url = "https://www.amazon.in/s?bbn=81107432031&rh=n%3A81107432031%2Cp_85%3A10440599031&_encoding=UTF8&content-id=amzn1.sym.58c90a12-100b-4a2f-8e15-7c06f1abe2be&pd_rd_r=b28b3a4c-f4d9-4493-95b4-8c24a6033448&pd_rd_w=OhXgB&pd_rd_wg=62wb4&pf_rd_p=58c90a12-100b-4a2f-8e15-7c06f1abe2be&pf_rd_r=81EXVS08S0QR0F33GBSB&ref=pd_hp_d_atf_unk"
    #url = "https://www.argos.co.uk/list/shop-for-apple-family-devices/?tag=ar:browse:clp:technology:m052:image:shop-for-apple-family-devices"
    url = "https://scrapeme.live/shop/"
    #url = "https://www.argos.co.uk/search/mobile/?clickOrigin=searchbar:cat:term:mobile"
    #url = "https://www.argos.co.uk/product/3100660?clickSR=slp:term:mobile:1:1057:1"

    
    #fields = ["About this product Details", "Reviews" ,"Questions and Answers","Bought Together"]
    fields = ["Product Name" ,"Price"]
    
    try:
        #fetch_and_process_pagination(url, fields)
        scrape_images(url)
    
    except Exception as e:
        print(f"An error occurred: {e}")


