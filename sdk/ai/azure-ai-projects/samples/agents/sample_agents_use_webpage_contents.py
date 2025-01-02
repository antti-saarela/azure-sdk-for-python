import os
import json
import time
import requests
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential


def write_to_file(data, filename):
    """Writes data to a file, converting it to a string first."""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(str(data))
        print(f"Data successfully written to {filename}")
    except Exception as e:
        print(f"Error writing to file {filename}: {e}")


def fetch_web_page_content(url):
    """Fetches the content of a web page."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching web page content: {e}")
        return None


def create_agent(project_client, model_name, agent_name, instructions):
    """Creates an agent with the given parameters."""
    try:
        agent = project_client.agents.create_agent(
            model=model_name,
            name=agent_name,
            instructions=instructions,
        )
        print(f"Agent created: {agent.id}")
        return agent
    except Exception as e:
        print(f"Error creating agent: {e}")
        return None


def create_thread(project_client):
    """Creates a new thread for communication."""
    try:
        thread = project_client.agents.create_thread()
        print(f"Thread created: {thread.id}")
        return thread
    except Exception as e:
        print(f"Error creating thread: {e}")
        return None


def create_message(project_client, thread_id, role, content):
    """Creates a message in the specified thread."""
    try:
        message = project_client.agents.create_message(
            thread_id=thread_id,
            role=role,
            content=content,
        )
        print(f"Message created: {message.id}")
        return message
    except Exception as e:
        print(f"Error creating message: {e}")
        return None


def process_run(project_client, thread_id, assistant_id):
    """Processes an agent run in the specified thread."""
    try:
        run = project_client.agents.create_and_process_run(
            thread_id=thread_id, assistant_id=assistant_id
        )
        print(f"Run status: {run.status}")
        if run.status == "failed":
            print(f"Run failed: {run.last_error}")
            return None
        return run
    except Exception as e:
        print(f"Error processing run: {e}")
        return None


def list_messages(project_client, thread_id):
    """Fetches all messages in the specified thread."""
    try:
        messages = project_client.agents.list_messages(thread_id=thread_id)
        return messages
    except Exception as e:
        print(f"Error listing messages: {e}")
        return None


def extract_assistant_reply(messages):
    """  
    Extracts the assistant's reply from the messages.  
    If 'text.value' exists, it is extracted. Otherwise, the entire response is returned.  
    """
    try:
        for message in messages.data:
            if message.role == "assistant":
                # Check if the content is a list and contains the expected structure
                if isinstance(message.content, list):
                    for item in message.content:
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "text"
                            and "text" in item
                            and "value" in item["text"]
                        ):
                            # Extract 'text.value'
                            return str(item["text"]["value"])
                # If the expected structure is not found, return the entire response
                return str(message.content)
        print("No assistant reply found.")
        return None
    except Exception as e:
        print(f"Error extracting assistant reply: {e}")
        return None


def main():
    try:
        # Initialize the AIProjectClient
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.environ["PROJECT_CONNECTION_STRING"],
        )

        # Fetch web page content
        url = "https://sosmeta.thl.fi/document-definitions/list/search"
        html_content = fetch_web_page_content(url)
        if not html_content:
            print("Failed to fetch web page content. Exiting.")
            return

        # Create a parsing agent
        parser_agent = create_agent(
            project_client,
            model_name=os.environ["AAA_MODEL_DEPLOYMENT_NAME"],
            agent_name="content-parser",
            instructions="Parse the provided HTML content and return links, titles, and descriptions in a well-structured format."
        )
        if not parser_agent:
            print("Failed to create parsing agent. Exiting.")
            return

        # Create a thread for the parsing agent
        thread = create_thread(project_client)
        if not thread:
            print("Failed to create thread for parsing agent. Exiting.")
            return

        # Send the HTML content to the parsing agent
        message = create_message(
            project_client, thread.id, "user", html_content)
        if not message:
            print("Failed to send message to parsing agent. Exiting.")
            return

        # Process the parsing agent run
        run = process_run(project_client, thread.id, parser_agent.id)
        if not run:
            print("Failed to process parsing agent run. Exiting.")
            return

        # Fetch and save the assistant's reply
        messages = list_messages(project_client, thread.id)
        if not messages:
            print("Failed to fetch messages from parsing agent. Exiting.")
            return

        assistant_reply = extract_assistant_reply(messages)
        if not assistant_reply:
            print("No reply from parsing agent. Exiting.")
            return

        # Write the assistant's reply to a file
        write_to_file(assistant_reply, "parsing_agent_reply.txt")

        # Use the assistant's reply as input for the next agent
        selector_agent = create_agent(
            project_client,
            model_name=os.environ["AAA_MODEL_DEPLOYMENT_NAME"],
            agent_name="link-selector",
            instructions="Valitse oikea linkki annettujen ohjeiden perusteella."
        )
        if not selector_agent:
            print("Failed to create link selector agent. Exiting.")
            return

        # Create a thread for the selector agent
        thread = create_thread(project_client)
        if not thread:
            print("Failed to create thread for link selector agent. Exiting.")
            return

        # Send the assistant's reply to the selector agent
        message_content = {
            "prompt": "Valitse ja palauta linkki, joka parhaiten vastaa erityishuollon hakemuksen täyttämisen ohjeita.",
            "links": assistant_reply
        }
        message = create_message(
            project_client, thread.id, "user", json.dumps(message_content)
        )
        if not message:
            print("Failed to send message to link selector agent. Exiting.")
            return

        # Process the selector agent run
        run = process_run(project_client, thread.id, selector_agent.id)
        if not run:
            print("Failed to process link selector agent run. Exiting.")
            return

        # Fetch and save the selector agent's reply
        messages = list_messages(project_client, thread.id)
        if not messages:
            print("Failed to fetch messages from link selector agent. Exiting.")
            return

        assistant_reply = extract_assistant_reply(messages)
        if not assistant_reply:
            print("No reply from link selector agent. Exiting.")
            return

        # Write the assistant's reply to a file
        write_to_file(assistant_reply, "link_selector_agent_reply.txt")

        # Cleanup agents
        project_client.agents.delete_agent(parser_agent.id)
        project_client.agents.delete_agent(selector_agent.id)
        print("Agents deleted successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
