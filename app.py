#imports
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import requests
import json
import os
import html2text
from langchain.chat_models import ChatOpenAI
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter
from langchain.prompts import ChatPromptTemplate
from llama_index import VectorStoreIndex
import openai
import markdown2

load_dotenv()
browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

#scrape raw HTML

def scrape_website(url: str):
    print("Scraping...")
    #headers for request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    #request data
    data = {
        "url": url,
        "elements": [{
            "selector": "body"
        }]
    }

    data_json = json.dumps(data) #python object to json

    # POST request to API
    response = requests.post(
        f"https://chrome.browserless.io/scrape?token={browserless_api_key}",
        headers = headers,
        data = data_json,
    )

    if response.status_code == 200:
        #Decode & Load string
        result = response.content
        data_str = result.decode('utf-8')
        data_dict = json.loads(data_str)

        html_string = data_dict['data'][0]['results'][0]['html']
        return html_string
    else:
        print(f"HTTP request failed. Status Code: {response.status_code}")

def get_base_url(url):
    parsed_url = urlparse(url)

    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    return base_url


# Turn relative url to absolute url in html (may need tweaks depending on site structure)

def convert_to_absolute_url(html, base_url):
    soup = BeautifulSoup(html, 'html.parser')

    for img_tag in soup.find_all('img'):
        if img_tag.get('src'):
            src = img_tag.get('src')
            if src.startswith(('http://', 'https://')):
                continue
            absolute_url = urljoin(base_url, src)
            img_tag['src'] = absolute_url
        elif img_tag.get('data-src'):
            src = img_tag.get('data-src')
            if src.startswith(('http://', 'https://')):
                continue
            absolute_url = urljoin(base_url, src)
            img_tag['data-src'] = absolute_url

    for link_tag in soup.find_all('a'):
        href = link_tag.get('href')
        if href.startswith(('http://', 'https://')):
            continue
        absolute_url = urljoin(base_url, href)
        link_tag['href'] = absolute_url

    updated_html = str(soup)

    return updated_html

#Convert html to markdown
def convert_html_to_markdown(html): 
    #initialize html2text converter
    converter = html2text.HTML2Text()
    converter.ignore_links = False
    markdown = converter.handle(html) #convert to markdown
    return markdown

def get_markdown_from_url(url):
    base_url = get_base_url(url)
    html = scrape_website(url)
    updated_html = convert_to_absolute_url(html, base_url)
    markdown = convert_html_to_markdown(updated_html)

    return markdown


#Create vector index from markdown
def create_index_from_text(markdown):
    text_splitter = TokenTextSplitter(
        separator="\n",
        chunk_size=1024,
        chunk_overlap=20,
        backup_separators=["\n\n", ".", ","]
    )
    node_parser = SimpleNodeParser(text_splitter=text_splitter).from_defaults(
        chunk_size=1024,
        chunk_overlap=20
    )
    nodes = node_parser.get_nodes_from_documents(
        [Document(text=markdown)], show_progress=True
    )

    index = VectorStoreIndex(nodes)

    print("Index Created")

    return index

#RAG

def generate_answer(query, index):
    #Get relevant data
    retriever = index.as_retriever()
    nodes = retriever.retrieve(query)
    texts = [node.node.text for node in nodes]

    print("Retrieved texts:", texts)

    #Generate OpenAI Answer
    model = ChatOpenAI(model_name = "gpt-3.5-turbo-16k-0613")
    template = """
    CONTEXT: {docs}
    You are a helpful teacher's assistant, above is some context, 
    Please answer the question, and make sure you follow ALL of the rules below:
    1. Answer the questions only based on context provided, do not make things up
    2. Answer questions in a helpful manner that is straight to the point, with clear structure & all relevant information that might help users answer the question
    3. Answer should be formatted in Markdown
    4. If there are relevant images, video, links, they are very important reference data, please include them as part of the answer

    QUESTION: {query}
    ANSWER (formatted in Markdown):
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    response = chain.invoke({"docs": texts, "query": query})

    return response.content


######### Markdown to HTML


def generate_html_page(markdown_content):
    html_content = markdown2.markdown(markdown_content)
    with open("template.html", "r") as template_file:
        html_template = template_file.read()
    # Insert the Markdown content into the HTML template
    final_html = html_template.replace("<!-- Markdown content will be inserted here -->", html_content)
    # Save the final HTML to a file
    with open("output.html", "w") as output_file:
        output_file.write(final_html)

##########

import webbrowser

def preview_markdown(markdown_content):
    # Create a temporary Markdown file
    with open('temp.md', 'w', encoding='utf-8') as temp_file:
        temp_file.write(markdown_content)

    # Specify the path to the Markdown file
    markdown_file_path = 'temp.md'

    # Open the Markdown file in the default web browser
    webbrowser.open(markdown_file_path, new=2)

url = "https://openstax.org/books/college-physics/pages/9-1-the-first-condition-for-equilibrium"
query = "What is an example of an object in static equilibrium and how do I calculate it?"
markdown = get_markdown_from_url(url)
index = create_index_from_text(markdown)
answer = generate_answer(query, index)
preview_markdown(answer)



