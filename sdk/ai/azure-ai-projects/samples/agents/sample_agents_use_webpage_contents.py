# flake8: noqa: E501
import os
import json
import re
import requests
import yaml
from datetime import datetime, timedelta
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MessageTextContent, MessageTextDetails, ThreadMessage
from azure.identity import DefaultAzureCredential
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def write_to_file(data, filename, output_dir="output", sub_dir=None, date=None):
    """Writes data to a file, converting it to a string first."""
    try:
        if sub_dir:
            dir_path = os.path.join(output_dir, sub_dir)
        else:
            dir_path = output_dir
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        filepath = os.path.join(dir_path, filename)
        with open(filepath, 'w', encoding='utf-8') as file:
            if date:
                file.write(f"timestamp_created: {date}\n")
            file.write(str(data))
        logging.info(f"Data successfully written to {filepath}")
    except Exception as e:
        logging.error(f"Error writing to file {filepath}: {e}")


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
        logging.error(f"Error fetching web page content: {e}")
        return None


def create_agent(project_client, model_name, agent_name, instructions):
    """Creates an agent with the given parameters."""
    try:
        agent = project_client.agents.create_agent(
            model=model_name,
            name=agent_name,
            instructions=instructions,
        )
        return agent
    except Exception as e:
        logging.error(f"Error creating agent: {e}")
        return None


def create_thread(project_client):
    """Creates a new thread for communication."""
    try:
        thread = project_client.agents.create_thread()
        logging.info(f"Thread created: {thread.id}")
        return thread
    except Exception as e:
        logging.error(f"Error creating thread: {e}")
        return None


def create_message(project_client, thread_id, role, content):
    """Creates a message in the specified thread."""
    try:
        message = project_client.agents.create_message(
            thread_id=thread_id,
            role=role,
            content=content,
        )
        logging.info(f"Message created: {message.id}")
        return message
    except Exception as e:
        logging.error(f"Error creating message: {e}")
        return None


def process_run(project_client, thread_id, assistant_id):
    """Processes an agent run in the specified thread."""
    try:
        run = project_client.agents.create_and_process_run(
            thread_id=thread_id, assistant_id=assistant_id
        )
        logging.info(f"Run status: {run.status}")
        if run.status == "failed":
            logging.error(f"Run failed: {run.last_error}")
            return None
        return run
    except Exception as e:
        logging.error(f"Error processing run: {e}")
        return None


def list_messages(project_client, thread_id):
    """Fetches all messages in the specified thread."""
    try:
        messages = project_client.agents.list_messages(thread_id=thread_id)
        return messages
    except Exception as e:
        logging.error(f"Error listing messages: {e}")
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
        logging.info("No 'text.value' found in assistant's reply.")
        return None
    except Exception as e:
        logging.error(f"Error extracting assistant reply: {e}")
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
            logging.info("No relative path found in the message content.")
            return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None


def process_topic(project_client, base_url, url_postfix, output_base_dir, topic, topic_links_yaml, selector_agent, form_filler_agent, form_filler_thread):
    """Processes a topic by creating agents, threads, and handling messages."""
    try:
        # Create new thread for the selector agent
        selector_thread = create_thread(project_client)
        if not selector_thread:
            logging.error(
                "Failed to create thread for selector agent. Skipping topic.")
            return

        # Send parser response to selector
        message_content = {
            "prompt": f"Valitse ja palauta vain suhteellinen linkki, joka parhaiten vastaa ohjeita lomakkeen täyttämiseksi, kun aiheena on {topic}. Palauta vain linkki, ei mitään muuta.",
            "links": topic_links_yaml['links']
        }
        message = create_message(
            project_client, selector_thread.id, "user", json.dumps(message_content))
        if not message:
            logging.error(
                "Failed to send message to selector. Skipping topic.")
            project_client.agents.delete_thread(selector_thread.id)
            return

        # Process selector run
        run = process_run(project_client, selector_thread.id,
                          selector_agent.id)
        if not run:
            logging.error("Failed to process selector run. Skipping topic.")
            project_client.agents.delete_thread(selector_thread.id)
            return

        # Fetch and save selector response
        messages = list_messages(project_client, selector_thread.id)
        if not messages:
            logging.error(
                "Failed to fetch messages from selector. Skipping topic.")
            project_client.agents.delete_thread(selector_thread.id)
            return
        selected_link = extract_assistant_reply(messages)
        if not selected_link:
            logging.error("No response from selector. Skipping topic.")
            project_client.agents.delete_thread(selector_thread.id)
            return
        write_to_file(
            selected_link, f"{topic}_selected_link.txt", output_dir=output_base_dir)
        relative_path = parse_relative_path_with_regex(selected_link)
        if not relative_path:
            logging.error("No relative path to instructions. Skipping topic.")
            project_client.agents.delete_thread(selector_thread.id)
            return

        logging.info(f"relative_path {relative_path}")

        # Delete the selector thread as it's no longer needed
        project_client.agents.delete_thread(selector_thread.id)

        # Convert relative link to absolute
        if not relative_path.startswith("http"):
            selected_link = f"{base_url}{relative_path}{url_postfix}"
        else:
            selected_link = relative_path
        write_to_file(
            selected_link, f"{topic}_combined_link.txt", output_dir=output_base_dir)

        # Fetch selected link content
        selected_page_content = fetch_web_page_content(selected_link)
        if not selected_page_content:
            logging.error(
                "Failed to fetch content of selected link. Skipping topic.")
            return
        write_to_file(selected_page_content,
                      f"{topic}_instructions_content.json", output_dir=output_base_dir)

        # Send selected link content to form filler with updated instructions
        user_input = selected_page_content + \
            "\n\n Käytä ohjeiden rakennetta ja palauta kaikki osiot täytettyinä keksityillä esimerkeillä. Varmista, että tiedot ovat johdonmukaisia tämän keskustelun aiempien lomakkeiden kanssa, säilyttäen nimet, yhteystiedot ja muut tiedot lomakkeiden välillä aiheen sisällä. Tee melko pitkiä kuvauksia eloisalla mutta realistisella sisällöllä tekstikentissä."
        message = create_message(
            project_client, form_filler_thread.id, "user", user_input
        )
        if not message:
            logging.error(
                "Failed to send message to form filler. Skipping topic.")
            return

        # Process form filler run
        run = process_run(
            project_client, form_filler_thread.id, form_filler_agent.id)
        if not run:
            logging.error("Failed to process form filler run. Skipping topic.")
            return

        # Fetch and save form filler response
        messages = list_messages(project_client, form_filler_thread.id)
        if not messages:
            logging.error(
                "Failed to fetch messages from form filler. Skipping topic.")
            return
        filled_form = extract_assistant_reply(messages)
        if not filled_form:
            logging.error("No response from form filler. Skipping topic.")
            return
        filled_form = filled_form.replace("```json", "").replace("```", "")
        write_to_file(filled_form, f"{topic}_filled_form.json",
                      output_dir=output_base_dir, sub_dir="filled_forms")

        return filled_form
    except Exception as e:
        logging.error(f"Error processing topic '{topic}': {e}")
        return None


def get_file_creation_time_from_yaml(filepath):
    """Gets the creation date from the YAML file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = yaml.safe_load(file)
            if 'timestamp_created' in content:
                return content['timestamp_created']
            return None
    except Exception as e:
        logging.error(f"Error getting file creation time from YAML: {e}")
        return None


def is_file_older_than_a_week(filepath):
    """Checks if the file is older than a week."""
    try:
        file_creation_date = get_file_creation_time_from_yaml(filepath)
        if file_creation_date:
            return datetime.now().date() - file_creation_date > timedelta(weeks=1)
        return False
    except Exception as e:
        logging.error(f"Error checking file age: {e}")
        return False


def read_yaml_file(filepath):
    """Reads a YAML file and returns its content."""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error reading YAML file {filepath}: {e}")
        return None


def main():
    """Main function to execute the process."""
    try:
        # Initialize AIProjectClient
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.environ["PROJECT_CONNECTION_STRING"],
        )
        base_url = "https://sosmeta.thl.fi"
        url_postfix = "/schema"
        model_name = os.environ["AAA_MODEL_DEPLOYMENT_NAME"]
        topics = [
            "# Adoption process",
            "Adoptionhakijoiden taustaselvitys",
            "Adoptioneuvonnan suunnitelma",
            "Adoptoitavan lapsen taustatiedot",
            "Adoptioneuvonnan lausunto",
            "Ilmoitus adoptionhakijoiden olosuhteiden muutoksesta",
            "# Child protection and welfare",
            "Ennakollinen lastensuojeluilmoitus",
            "Ennakollisen lastensuojeluilmoituksen arvio",
            "Huostaanottohakemus hallinto-oikeudelle",
            "Hoito- ja kasvatussuunnitelma",
            "Ilmoitus hyvinvointialueen alueelle sijoitetusta lapsesta",
            "# Employment and social services",
            "Aktivointisuunnitelma",
            "Ilmoitus kuntouttavasta työtoiminnasta",
            "Asia-asiakirja",
            "Asiakkuusasiakirja",
            "Asiakkaan yksilöllinen varautumissuunnitelma",
            "# Family and parental matters",
            "Avioliiton aikana syntyneen tai raskausaikana tunnustetun lapsen vanhempien neuvotteluasiakirja vanhemmuuden selvittämiseksi",
            "Hyväksymisasiakirja vanhemmuusasiassa",
            "Ilmoitus lapsen elatusavun lakimääräisestä muutoksesta",
            "Ilmoitus lapsen huoltajan tarpeesta",
            "# Special care and health services",
            "Hakemus erityishuoltoon",
            "Hakemus hallinto-oikeudelle lapsen tutkimiseksi",
            "Hakemus päihde- ja riippuvuustyön erityiseen palveluun",
            "Ilmoitus aluehallintovirastolle asiakkaan sitomisesta",
            "# Safety and risk assessment",
            "Häirinnän ja vainon riskiarvio",
            "Ilmoitus sosiaalihuollon tarpeesta",
        ]

        # Prepare output directory with timestamp
        timestamp_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        output_base_dir = os.path.join("output", timestamp_str)

        # Check if parser_response.yaml exists and is older than a week
        filepath = os.path.join("output", "parser_response.yaml")
        if os.path.exists(filepath) and not is_file_older_than_a_week(filepath):
            logging.info("parser_response.yaml is fresh. Using existing file.")
            topic_links_yaml = read_yaml_file(filepath)
            if not topic_links_yaml or len(topic_links_yaml['links']) == 0:
                logging.error(
                    "Failed to read or empty parser_response.yaml. Exiting.")
                return
        else:
            # Fetch web page content once
            url = f"{base_url}/document-definitions/list/search"
            html_content = fetch_web_page_content(url)
            if not html_content:
                logging.error("Failed to fetch web page content. Exiting.")
                return

            # Create parser agent
            parser_agent = create_agent(
                project_client,
                model_name=model_name,
                agent_name="content-parser",
                instructions="Jäsennä annettu HTML-sisältö ja palauta linkit, otsikot ja kuvaukset, jotka liittyvät asiakirjojen täyttämiseen. Palauta vain linkit, tila, otsikot ja kuvaukset hyvin jäsennellyssä koneellisesti luettavassa muodossa, kuten YAML."
            )
            if not parser_agent:
                logging.error(
                    "Failed to create content parser agent. Exiting.")
                return

            # Create thread for parser
            parser_thread = create_thread(project_client)
            if not parser_thread:
                logging.error("Failed to create thread for parser. Exiting.")
                return

            # Send HTML content to parser
            message = create_message(
                project_client, parser_thread.id, "user", html_content)
            if not message:
                logging.error("Failed to send message to parser. Exiting.")
                return

            # Process parser run
            run = process_run(
                project_client, parser_thread.id, parser_agent.id)
            if not run:
                logging.error("Failed to process parser run. Exiting.")
                return

            # Fetch and save parser response
            messages = list_messages(project_client, parser_thread.id)
            if not messages:
                logging.error("Failed to fetch messages from parser. Exiting.")
                return
            topic_links_yaml = extract_assistant_reply(messages)
            if not topic_links_yaml or not topic_links_yaml.strip():
                logging.error(
                    "No response from parser or response is empty. Exiting.")
                return
            topic_links_yaml = topic_links_yaml.replace(
                "```yaml", "").replace("```", "")
            date = datetime.now().strftime('%Y-%m-%d')
            write_to_file(topic_links_yaml, "parser_response.yaml", date)

            # Clean up parser agent and thread
            project_client.agents.delete_agent(parser_agent.id)
            project_client.agents.delete_thread(parser_thread.id)

        selector_agent = None
        form_filler_agent = None
        form_filler_thread = None

        for topic in topics:
            if topic.startswith("#"):
                # Reset form filler agent and thread when a new topic group starts
                if form_filler_agent:
                    project_client.agents.delete_agent(form_filler_agent.id)
                if form_filler_thread:
                    project_client.agents.delete_thread(form_filler_thread.id)

                # Create new form filler agent and thread
                form_filler_agent = create_agent(
                    project_client,
                    model_name=model_name,
                    agent_name="form-filler",
                    instructions=(
                        "Käytä annettua JSON-sisältöä täyttöohjeina ja täytä uusi lomake keksityillä tiedoilla."
                        "Palauta vain täytetty lomake. Täytä kaikki ohjeissa mainitut osiot, myös ne, joita ei ole merkitty pakollisiksi."
                        "Täytä kaikki hierarkkiset tasot. Keksi selityksiä ja tekstejä vapaatekstikenttiin tarvittaessa."
                        "Varmista, että käytetyt tiedot ovat johdonmukaisia tämän keskustelun aiempien lomakkeiden kanssa, säilyttäen nimet, yhteystiedot ja muut tiedot lomakkeiden välillä aiheen sisällä."
                        "Jos ja kun keksit uusia nimiä, vältä kaikkein yleisimpiä sukunimiä ja etunimiä. Ole huolellinen käyttäessäsi nimiä keskusteluhistoriasta."
                        "Ole tarkka syntymäaikojen ja sosiaaliturvatunnuksien luomsen kanssa, jotta ne olisivat yhdenmukaisia."
                        "Käytä huolellista ja hyvää suomenkieltä."
                        "Palauta täytetty lomake JSON-muodossa."
                    )
                )
                if not form_filler_agent:
                    logging.error(
                        "Failed to create form filler agent. Exiting.")
                    return

                form_filler_thread = create_thread(project_client)
                if not form_filler_thread:
                    logging.error("Failed to create thread. Exiting.")
                    return

                # Create new selector agent for the topic group
                if selector_agent:
                    project_client.agents.delete_agent(selector_agent.id)

                selector_agent = create_agent(
                    project_client,
                    model_name=model_name,
                    agent_name="link-selector",
                    instructions="Valitse oikea linkki annettujen ohjeiden perusteella."
                )
                if not selector_agent:
                    logging.error(
                        "Failed to create link selector agent. Exiting.")
                    return

                continue

            process_topic(
                project_client, base_url, url_postfix, output_base_dir, topic, topic_links_yaml, selector_agent, form_filler_agent, form_filler_thread)

        # Clean up agents and threads after processing all topics
        if selector_agent:
            project_client.agents.delete_agent(selector_agent.id)
        if form_filler_agent:
            project_client.agents.delete_agent(form_filler_agent.id)
        if form_filler_thread:
            project_client.agents.delete_thread(form_filler_thread.id)

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
