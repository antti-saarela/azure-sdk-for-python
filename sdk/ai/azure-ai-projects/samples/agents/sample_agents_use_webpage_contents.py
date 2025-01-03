# flake8: noqa: E501

import os
import json
import re
import requests
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MessageTextContent, MessageTextDetails, ThreadMessage
from azure.identity import DefaultAzureCredential


def write_to_file(data, filename):
    """Writes data to a file, converting it to a string first."""
    try:
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(str(data))
        print(f"Data successfully written to {filepath}")
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
    """Extracts the assistant's reply from the messages."""
    try:
        for message in messages.data:
            if isinstance(message, ThreadMessage) and message.role == "assistant":
                if isinstance(message.content, list):
                    if len(message.content) > 0 and isinstance(message.content[0], MessageTextContent):
                        first_content = message.content[0]
                        if (
                            isinstance(first_content, MessageTextContent) and
                            first_content.type == "text" and
                            hasattr(first_content, "text") and
                            isinstance(first_content.text, MessageTextDetails) and
                            hasattr(first_content.text, "value")
                        ):
                            return str(first_content.text.value)
        print("No 'text.value' found in assistant's reply.")
        return None
    except Exception as e:
        print(f"Error extracting assistant reply: {e}")
        return None


def parse_relative_path_with_regex(message_content):
    """Parses the relative path from the agent's reply using regex."""
    try:
        match = re.search(
            r'(?:link)?:?\s*["\']?(\/document-definitions\/[^\s"\'\\]+)', message_content)
        if match:
            result = match.group(1)
            return result.rstrip("'")
        else:
            print("No relative path found in the message content.")
            return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def process_topic(project_client, base_url, url_postfix, topic, model_name):
    try:
        # Fetch web page content
        url = f"{base_url}/document-definitions/list/search"
        html_content = fetch_web_page_content(url)
        if not html_content:
            print("Failed to fetch web page content. Skipping topic.")
            return

        # Create parser agent
        parser_agent = create_agent(
            project_client,
            model_name=model_name,
            agent_name="content-parser",
            instructions="Parse the given HTML content and return the links, titles, and descriptions related to document filling. Return only the links, status, titles, and descriptions in a well-structured machine-readable format, such as YAML."
        )
        if not parser_agent:
            print("Failed to create content parser agent. Skipping topic.")
            return

        # Create thread for parser
        thread = create_thread(project_client)
        if not thread:
            print("Failed to create thread for parser. Skipping topic.")
            return

        # Send HTML content to parser
        message = create_message(
            project_client, thread.id, "user", html_content)
        if not message:
            print("Failed to send message to parser. Skipping topic.")
            return

        # Process parser run
        run = process_run(project_client, thread.id, parser_agent.id)
        if not run:
            print("Failed to process parser run. Skipping topic.")
            return

        # Fetch and save parser response
        messages = list_messages(project_client, thread.id)
        if not messages:
            print("Failed to fetch messages from parser. Skipping topic.")
            return
        assistant_reply = extract_assistant_reply(messages)
        if not assistant_reply:
            print("No response from parser. Skipping topic.")
            return
        assistant_reply = assistant_reply.replace(
            "```yaml", "").replace("```", "")
        write_to_file(assistant_reply, f"{topic}_parser_response.yaml")

        # Create selector agent
        selector_agent = create_agent(
            project_client,
            model_name=model_name,
            agent_name="link-selector",
            instructions="Select the correct link based on the given instructions."
        )
        if not selector_agent:
            print("Failed to create link selector agent. Skipping topic.")
            return

        # Create thread for selector
        thread = create_thread(project_client)
        if not thread:
            print("Failed to create thread for selector. Skipping topic.")
            return

        # Send parser response to selector
        message_content = {
            "prompt": f"Select and return only the relative link that best matches the instructions for filling out the form {topic}. Return only the link, nothing else.",
            "links": assistant_reply
        }
        message = create_message(
            project_client, thread.id, "user", json.dumps(message_content))
        if not message:
            print("Failed to send message to selector. Skipping topic.")
            return

        # Process selector run
        run = process_run(project_client, thread.id, selector_agent.id)
        if not run:
            print("Failed to process selector run. Skipping topic.")
            return

        # Fetch and save selector response
        messages = list_messages(project_client, thread.id)
        if not messages:
            print("Failed to fetch messages from selector. Skipping topic.")
            return
        selected_link = extract_assistant_reply(messages)
        if not selected_link:
            print("No response from selector. Skipping topic.")
            return
        write_to_file(selected_link, f"{topic}_selected_link.txt")

        relative_path = parse_relative_path_with_regex(selected_link)
        if not relative_path:
            print("No relative path to instructions. Skipping topic.")
            return
        print(f"relative_path {relative_path}")

        # Convert relative link to absolute
        if not relative_path.startswith("http"):
            selected_link = f"{base_url}{relative_path}{url_postfix}"
        else:
            selected_link = relative_path
        write_to_file(selected_link, f"{topic}_combined_link.txt")

        # Fetch selected link content
        selected_page_content = fetch_web_page_content(selected_link)
        if not selected_page_content:
            print("Failed to fetch content of selected link. Skipping topic.")
            return
        write_to_file(selected_page_content,
                      f"{topic}_instructions_content.json")

        # Create form filler agent
        form_filler_agent = create_agent(
            project_client,
            model_name=model_name,
            agent_name="form-filler",
            instructions=("Use the given HTML content as filling instructions and fill out a new form with invented data."
                          "Return only the filled form. Fill in all the sections mentioned in the instructions, even those not marked as mandatory."
                          "Fill in all hierarchical levels. Invent explanations and texts for free text fields if necessary. Return the filled form in JSON format.")
        )
        if not form_filler_agent:
            print("Failed to create form filler agent. Skipping topic.")
            return

        # Create thread for form filler
        thread = create_thread(project_client)
        if not thread:
            print("Failed to create thread for form filler. Skipping topic.")
            return

        # Send selected link content to form filler
        message = create_message(
            project_client, thread.id, "user", selected_page_content +
            "\n\n Use the structure of the instructions and return all sections filled with invented examples. Make quite lengthy descriptions with vivid but realistic content in text fields."
        )
        if not message:
            print("Failed to send message to form filler. Skipping topic.")
            return

        # Process form filler run
        run = process_run(project_client, thread.id, form_filler_agent.id)
        if not run:
            print("Failed to process form filler run. Skipping topic.")
            return

        # Fetch and save form filler response
        messages = list_messages(project_client, thread.id)
        if not messages:
            print("Failed to fetch messages from form filler. Skipping topic.")
            return
        filled_form = extract_assistant_reply(messages)
        if not filled_form:
            print("No response from form filler. Skipping topic.")
            return
        filled_form = filled_form.replace("```json", "").replace("```", "")
        write_to_file(filled_form, f"{topic}_filled_form.json")

        # Clean up agents
        project_client.agents.delete_agent(parser_agent.id)
        project_client.agents.delete_agent(selector_agent.id)
        project_client.agents.delete_agent(form_filler_agent.id)
        print("Agents successfully deleted.")
    except Exception as e:
        print(f"Error processing topic '{topic}': {e}")


def main():
    try:
        # Initialize AIProjectClient
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.environ["PROJECT_CONNECTION_STRING"],
        )

        base_url = "https://sosmeta.thl.fi"
        url_postfix = "/schema"
        model_name = os.environ["AAA_MODEL_DEPLOYMENT_NAME"]
        topics = ["ennakovia ilmoitus lastensuojelusta",
                  "Adoptioneuvonnan lausunto", "ilmotus kuntouttava työtoiminta"]
        for topic in topics:
            process_topic(project_client, base_url,
                          url_postfix, topic, model_name)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
