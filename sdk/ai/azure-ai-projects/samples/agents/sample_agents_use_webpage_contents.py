import re
import os
import json
import time
import requests
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import MessageTextContent, MessageTextDetails, ThreadMessage
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
    Always returns the 'text.value' field if it exists.
    """
    try:
        for message in messages.data:
            if isinstance(message, ThreadMessage) and message.role == "assistant":
                # Tarkistetaan, että content on sanakirja ja sisältää odotetun rakenteen
                if isinstance(message.content, list):
                    if len(message.content) > 0 and isinstance(message.content[0], MessageTextContent):
                        first_content = message.content[0]
                        if (
                            isinstance(first_content, MessageTextContent) and
                            first_content["type"] == "text" and
                            hasattr(first_content, "text") and
                            isinstance(first_content["text"], MessageTextDetails) and
                            hasattr(first_content["text"], "value")
                        ):
                            # Palautetaan 'text.value'
                            return str(first_content["text"].value)
        # Jos 'text.value' ei löydy, tulostetaan virheilmoitus
        print("No 'text.value' found in assistant's reply.")
        return None
    except Exception as e:
        print(f"Error extracting assistant reply: {e}")
        return None


def parse_relative_path_with_regex(message_content):
    """
    Parses the relative path from the agent's reply using regex.
    Handles cases where the link is:
    - A standalone relative path
    - Part of a key-value pair (e.g., "link" or "linkki")
    - With or without quotes
    - Removes trailing single quotes if present
    """
    try:
        # Etsitään linkki muodossa:
        # 1. "link": "/document-definitions/..."
        # 2. linkki: /document-definitions/...
        # 3. "/document-definitions/..." (lainausmerkeissä)
        # 4. /document-definitions/... (pelkkä polku)
        match = re.search(
            r'(?:link|linkki)?:?\s*["\']?(\/document-definitions\/[^\s"\'\\]+)', message_content)
        if match:
            # Palautetaan löydetty linkki ja poistetaan mahdollinen lopussa oleva yksittäinen lainausmerkki
            result = match.group(1)
            return result.rstrip("'")
        else:
            print("No relative path found in the message content.")
            return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


def main():
    try:
        # Alustetaan AIProjectClient
        project_client = AIProjectClient.from_connection_string(
            credential=DefaultAzureCredential(),
            conn_str=os.environ["PROJECT_CONNECTION_STRING"],
        )

        # Haetaan verkkosivun sisältö
        base_url = "https://sosmeta.thl.fi"
        url = f"{base_url}/document-definitions/list/search"
        html_content = fetch_web_page_content(url)
        if not html_content:
            print("Verkkosivun sisällön haku epäonnistui. Lopetetaan.")
            return

        # Luodaan agentti HTML-sisällön jäsentämiseen
        parser_agent = create_agent(
            project_client,
            model_name=os.environ["AAA_MODEL_DEPLOYMENT_NAME"],
            agent_name="sisällön-jäsentäjä",
            instructions="Jäsennä annettu HTML-sisältö ja palauta sen sisältämät linkit, otsikot ja kuvaukset, joitka liittyvät dokumenttien täyttämiseen. Palauta vain linkit, tila, otsikot ja kuvaukset hyvin jäsennellyssä koneluettavassa muodossa, kuten YAMLina."
        )
        if not parser_agent:
            print("Sisällön jäsentäjän luominen epäonnistui. Lopetetaan.")
            return

        # Luodaan keskusteluketju jäsentäjälle
        thread = create_thread(project_client)
        if not thread:
            print("Keskusteluketjun luominen jäsentäjälle epäonnistui. Lopetetaan.")
            return

        # Lähetetään HTML-sisältö jäsentäjälle
        message = create_message(
            project_client, thread.id, "user", html_content)
        if not message:
            print("Viestin lähettäminen jäsentäjälle epäonnistui. Lopetetaan.")
            return

        # Prosessoidaan jäsentäjän suoritus
        run = process_run(project_client, thread.id, parser_agent.id)
        if not run:
            print("Jäsentäjän suorituksen prosessointi epäonnistui. Lopetetaan.")
            return

        # Haetaan ja tallennetaan jäsentäjän vastaus
        messages = list_messages(project_client, thread.id)
        if not messages:
            print("Viestien haku jäsentäjältä epäonnistui. Lopetetaan.")
            return
        assistant_reply = extract_assistant_reply(messages)
        if not assistant_reply:
            print("Ei vastausta jäsentäjältä. Lopetetaan.")
            return
        write_to_file(assistant_reply, "jäsentäjän_vastaus.txt")

        # Käytetään jäsentäjän vastausta seuraavalle agentille
        selector_agent = create_agent(
            project_client,
            model_name=os.environ["AAA_MODEL_DEPLOYMENT_NAME"],
            agent_name="linkin-valitsija",
            instructions="Valitse oikea linkki annettujen ohjeiden perusteella."
        )
        if not selector_agent:
            print("Linkin valitsijan luominen epäonnistui. Lopetetaan.")
            return

        # Luodaan keskusteluketju valitsijalle
        thread = create_thread(project_client)
        if not thread:
            print("Keskusteluketjun luominen valitsijalle epäonnistui. Lopetetaan.")
            return

        topic = "ennakovia ilmoitus lastensuojelusta"
        # Lähetetään jäsentäjän vastaus valitsijalle
        message_content = {
            "prompt": f"Valitse ja palauta vain sunteellinen linkki, joka parhaiten vastaa lomakkeen {topic} täyttämisen ohjeita. Palauta vain linkki, älä muuta",
            "links": assistant_reply
        }
        message = create_message(
            project_client, thread.id, "user", json.dumps(message_content)
        )
        if not message:
            print("Viestin lähettäminen valitsijalle epäonnistui. Lopetetaan.")
            return

        # Prosessoidaan valitsijan suoritus
        run = process_run(project_client, thread.id, selector_agent.id)
        if not run:
            print("Valitsijan suorituksen prosessointi epäonnistui. Lopetetaan.")
            return

        # Haetaan ja tallennetaan valitsijan vastaus
        messages = list_messages(project_client, thread.id)
        if not messages:
            print("Viestien haku valitsijalta epäonnistui. Lopetetaan.")
            return
        selected_link = extract_assistant_reply(messages)
        if not selected_link:
            print("Ei vastausta valitsijalta. Lopetetaan.")
            return
        write_to_file(selected_link, "suhteellinen_polku.txt")

        relative_path = parse_relative_path_with_regex(selected_link)
        if not relative_path:
            print("Ei suhteellista polkua ohjeeseen. Lopetetaan.")
            return

        print(f"relative_path {relative_path}")

        # Muutetaan suhteellinen linkki absoluuttiseksi
        if not relative_path.startswith("http"):
            selected_link = f"{base_url}{relative_path}"
        else:
            selected_link = relative_path
        write_to_file(selected_link, "yhdistetty_linkki.txt")

        # Haetaan valitun linkin takana oleva sivu
        selected_page_content = fetch_web_page_content(selected_link)
        if not selected_page_content:
            print("Valitun linkin sisällön haku epäonnistui. Lopetetaan.")
            return

        write_to_file(selected_page_content, "ohjeen_sisalto.html")

        # Luodaan agentti hakemuksen täyttämiseen
        form_filler_agent = create_agent(
            project_client,
            model_name=os.environ["AAA_MODEL_DEPLOYMENT_NAME"],
            agent_name="hakemuksen-täyttäjä",
            instructions="Käytä annettua HTML-sisältöä täyttöohjeena ja täytä uusi lomake keksityillä tiedoilla. Palauta vain täytetty lomake. Täytä kaikki ohjeessa olevat kohdat, myös ne, jotka eivät ole ohjeessa merkitty pakollisiksi. Täytä kaikkien hierarkiatasojen tiedot. Keksi tarvittaesaa selitteet ja tekstit vapaatekstikenttiin. Palauta täytetty lomake Markdown -muodossa."
        )
        if not form_filler_agent:
            print("Hakemuksen täyttäjän luominen epäonnistui. Lopetetaan.")
            return

        # Luodaan keskusteluketju täyttäjälle
        thread = create_thread(project_client)
        if not thread:
            print("Keskusteluketjun luominen täyttäjälle epäonnistui. Lopetetaan.")
            return

        # Lähetetään valitun linkin sisältö täyttäjälle
        message = create_message(
            project_client, thread.id, "user", selected_page_content +
            "\n\n Käytä ohjeen numerointia ja palauta kaikki numeroidut kohdat täytettyinä keksityillä esimerkeillä. Tee tekstikenttiin keskipitkiä kuvauksia."
        )
        if not message:
            print("Viestin lähettäminen täyttäjälle epäonnistui. Lopetetaan.")
            return

        # Prosessoidaan täyttäjän suoritus
        run = process_run(project_client, thread.id, form_filler_agent.id)
        if not run:
            print("Täyttäjän suorituksen prosessointi epäonnistui. Lopetetaan.")
            return

        # Haetaan ja tallennetaan täyttäjän vastaus
        messages = list_messages(project_client, thread.id)
        if not messages:
            print("Viestien haku täyttäjältä epäonnistui. Lopetetaan.")
            return
        filled_form = extract_assistant_reply(messages)
        if not filled_form:
            print("Ei vastausta täyttäjältä. Lopetetaan.")
            return

        filled_form = filled_form.replace("```markdown", "").replace("```", "")

        write_to_file(filled_form, "täytetty_hakemus.md")

        # Siivotaan agentit
        project_client.agents.delete_agent(parser_agent.id)
        project_client.agents.delete_agent(selector_agent.id)
        project_client.agents.delete_agent(form_filler_agent.id)
        print("Agentit poistettu onnistuneesti.")

    except Exception as e:
        print(f"Tapahtui virhe: {e}")


if __name__ == "__main__":
    main()
