# ------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ------------------------------------
"""DESCRIPTION:  
    This sample demonstrates how to use agent operations with the  
    OpenAPI tool from the Azure Agents service using a synchronous client.  
    To learn more about OpenAPI specs, visit https://learn.microsoft.com/openapi  
USAGE:  
    python sample_agents_openapi.py  
    Before running the sample:  
    pip install azure-ai-projects azure-identity jsonref  
    Set this environment variables with your own values:  
    PROJECT_CONNECTION_STRING - the Azure AI Project connection string, as found in your AI Foundry project.  
"""
import os
import json
import jsonref
import time
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.projects.models import OpenApiTool, OpenApiAnonymousAuthDetails

project_client = AIProjectClient.from_connection_string(
    credential=DefaultAzureCredential(),
    conn_str=os.environ["PROJECT_CONNECTION_STRING"],
)

# [START create_agent_with_openapi]
with open("./weather_openapi.json", "r") as f:
    openapi_spec = jsonref.loads(f.read())

# Create Auth object for the OpenApiTool (note that connection or managed identity auth setup requires additional setup in Azure)
auth = OpenApiAnonymousAuthDetails()

# Initialize agent OpenApi tool using the read in OpenAPI spec
openapi = OpenApiTool(
    name="get_weather", spec=openapi_spec, description="Retrieve weather information for a location", auth=auth
)

# Create agent with OpenApi tool and process assistant run
with project_client:
    start_time = time.time()  # Start the timer

    agent = project_client.agents.create_agent(
        model=os.environ["AAA_MINI_MODEL_NAME"], name="my-assistant", instructions="Autat säätilojen kanssa ja konvertoit Fahrenheit-asteet tarvittaessa Celcius-asteiksi.", tools=openapi.definitions
    )
    # [END create_agent_with_openapi]
    print(f"Created agent, ID: {agent.id}")

    # Create thread for communication
    thread = project_client.agents.create_thread()
    print(f"Created thread, ID: {thread.id}")

    # Create message to thread
    message = project_client.agents.create_message(
        thread_id=thread.id,
        role="user",
        content="Mikä on säätila Helsingissä nyt? Anna säätilan perusteella pukeutumissuosituksia ulkoiluun.",
    )
    print(f"Created message, ID: {message.id}")

    # Create and process agent run in thread with tools
    run = project_client.agents.create_and_process_run(
        thread_id=thread.id, assistant_id=agent.id)
    print(f"Run finished with status: {run.status}")

    end_time = time.time()  # End the timer
    execution_time = end_time - start_time  # Calculate the execution time
    print(f"Execution time: {execution_time} seconds")

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    # Delete the assistant when done
    project_client.agents.delete_agent(agent.id)
    print("Deleted agent")

    # Fetch and log all messages
    messages = project_client.agents.list_messages(thread_id=thread.id)
    print(f"Messages: {messages}")

    # Convert messages to a list of dictionaries
    messages_list = []
    for msg in messages.data:
        msg_dict = msg.as_dict()
        # Ensure text content is properly encoded in UTF-8
        if 'content' in msg_dict:
            for content in msg_dict['content']:
                if content['type'] == 'text' and 'text' in content:
                    content['text']['value'] = content['text']['value'].encode(
                        'utf-8').decode('utf-8')
        messages_list.append(msg_dict)

    # Write results to disk in JSON format
    results = {
        "agent_id": agent.id,
        "thread_id": thread.id,
        "message_id": message.id,
        "run_status": run.status,
        "execution_time": execution_time,
        "messages": messages_list
    }

    with open("results.json", "w", encoding='utf-8') as results_file:
        json.dump(results, results_file, ensure_ascii=False, indent=4)

    print("Results written to results.json")
