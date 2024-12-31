import os
import json
import time
import requests
from bs4 import BeautifulSoup
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import BingGroundingTool, MessageTextContent


def write_to_file(data, filename):
    """  
    Writes data to a JSON file with UTF-8 encoding.  
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Data successfully written to {filename}")
    except Exception as e:
        print(f"Failed to write to file {filename}: {e}")


def fetch_web_page_content(url):
    """  
    Fetches the content of a web page.  
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Failed to fetch web page content: {e}")
        return None


def parse_web_page_content(html_content):
    """  
    Parses the HTML content of a web page.  
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        print(f"Failed to parse web page content: {e}")
        return None


def main():
    try:
        # Initialize the AIProjectClient with the connection string and default credentials
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.environ["PROJECT_CONNECTION_STRING"],
        )

        # Retrieve the Bing connection ID
        bing_connection = project_client.connections.get(
            connection_name=os.environ["BING_CONNECTION_NAME"]
        )
        conn_id = bing_connection.id
        print(f"Bing Connection ID: {conn_id}")

        # Initialize the Bing grounding tool with the connection ID
        bing = BingGroundingTool(connection_id=conn_id)

        # Create an agent with the Bing grounding tool
        with project_client:
            agent = project_client.agents.create_agent(
                model=os.environ["AAA_MODEL_DEPLOYMENT_NAME"],
                name="my-assistant",
                instructions=(
                    "Autat sosiaalihuollon asiakirjojen etsimisessä internetistä osoitteen "
                    "termeta.thl.fi/document-definitions alta. Haet esimerkiksi Asia-asiakirjaa haulla "
                    "site:termeta.thl.fi/document-definitions Asia-asiakirja. Korvaa hakutermi lopussa "
                    "käyttäjän antamien tietojen perusteella."
                ),
                tools=bing.definitions,
                headers={"x-ms-enable-preview": "true"},
            )
            print(f"Created agent, ID: {agent.id}")

            # Create a thread for communication
            thread = project_client.agents.create_thread()
            print(f"Created thread, ID: {thread.id}")

            # Create a message in the thread
            message = project_client.agents.create_message(
                thread_id=thread.id,
                role="user",
                content=(
                    "Etsi 'Hakemus erityishuoltoon'. Palauta täsmällisesti ensimmäinen linkki hakutuloksista, joka on muotoa https://termeta.thl.fi/document-definitions/<ID>, jossa loppuosan ID vaihtuu, mutta perässä ei ole muuta"
                    "Palauta vastauksessasi myös hakutermit joita käytit haussa."
                ),
            )
            print(f"Created message, ID: {message.id}")

            # Create and process an agent run in the thread with tools
            run = project_client.agents.create_and_process_run(
                thread_id=thread.id, assistant_id=agent.id
            )
            print(f"Run finished with status: {run.status}")

            if run.status == "failed":
                print(f"Run failed: {run.last_error}")
                return

            # Fetch and log all messages
            messages = project_client.agents.list_messages(thread_id=thread.id)
            print(f"Messages: {messages}")

            # Write Bing grounding results to a file
            write_to_file([message.as_dict()
                          for message in messages.data], "bing_grounding_results.json")

            # Extract the URL from the assistant's response
            assistant_response = messages.data[0].content[0].text['annotations']
            url = None
            for annotation in assistant_response:
                if annotation['type'] == 'url_citation':
                    url = annotation['url_citation']['url']
                    break
            print(f"Extracted URL: {url}")

            # Fetch and parse the web page content
            if url:
                html_content = fetch_web_page_content(url)
                if html_content:
                    web_page_text = parse_web_page_content(html_content)
                    if web_page_text:
                        # Create a new agent for processing the web page content
                        content_agent = project_client.agents.create_agent(
                            model=os.environ["AAA_MODEL_DEPLOYMENT_NAME"],
                            name="sosmeta-assistant",
                            instructions="Olet avulias assistentti. Täytä lomake ohjeiden mukaan keksimälälsi esimerkkidatalla.",
                        )
                        print(f"Created content agent, ID: {content_agent.id}")

                        # Create a new thread for the content agent
                        thread = project_client.agents.create_thread()
                        print(
                            f"Created thread for content agent, ID: {thread.id}")

                        # Send the web page content and instructions to the content agent
                        message = project_client.agents.create_message(
                            thread_id=thread.id,
                            role="user",
                            content=web_page_text + "\n\nTäytä lomake ohjeiden mukaan."
                        )
                        print(
                            f"Created message for content agent, ID: {message.id}")

                        # Create and poll a run for the content agent
                        run = project_client.agents.create_run(
                            thread_id=thread.id, assistant_id=content_agent.id
                        )
                        while run.status in ["queued", "in_progress", "requires_action"]:
                            time.sleep(1)
                            run = project_client.agents.get_run(
                                thread_id=thread.id, run_id=run.id
                            )
                            print(f"Run status: {run.status}")

                        # Fetch and log all messages from the content agent
                        messages = project_client.agents.list_messages(
                            thread_id=thread.id)
                        content_agent_results = []
                        for data_point in reversed(messages.data):
                            last_message_content = data_point.content[-1]
                            if isinstance(last_message_content, MessageTextContent):
                                content_agent_results.append({
                                    "role": data_point.role,
                                    "text": last_message_content.text.value
                                })
                                print(
                                    f"{data_point.role}: {last_message_content.text.value}")

                        # Write content agent results to a file
                        write_to_file(content_agent_results,
                                      "content_agent_results.json")

                        # Delete the content agent when done
                        project_client.agents.delete_agent(content_agent.id)
                        print("Deleted content agent")

            # Delete the initial assistant when done
            project_client.agents.delete_agent(agent.id)
            print("Deleted initial agent")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
