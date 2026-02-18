import gradio as gr
import yaml
import os
from pathlib import Path
from ai_engine import get_ai_response  # Placeholder AI
from storage import load_config, save_config  # Encrypted storage
from observation import analyze_screen  # Stub for screen observation

BASE_DIR = Path(__file__).parent

# Load initial config (e.g., from encrypted storage)
config = load_config()

# Mock chat function (replace with real AI later)
def chat_handler(message, history):
    # Apply user config (e.g., style)
    style = config.get('response_style', 'concise')
    response = get_ai_response(message)  # Calls placeholder
    return f"{style.capitalize()} response: {response}"

# Function to update config
def update_config(new_yaml):
    try:
        new_config = yaml.safe_load(new_yaml)
        save_config(new_config)  # Encrypted save
        return "Config updated successfully!"
    except Exception as e:
        return f"Error: {str(e)}"

# Function for observation (with consent)
def trigger_observation(user_consent: bool):
    """Only proceed if the user has explicitly checked the consent box."""
    if not user_consent:
        return "Consent required. Check the box above to grant screen access."
    return analyze_screen()

# Plugin loader stub (list files in plugins/)
def list_plugins():
    plugins_dir = BASE_DIR / 'plugins'
    plugins = [f for f in os.listdir(plugins_dir) if f.endswith('.py')]
    return "\n".join(plugins) if plugins else "No plugins yet. Add Python files to ./plugins/"

# Build the UI
with gr.Blocks(title="LocalAI Copilot") as app:
    gr.Markdown("# LocalAI Copilot - UI Prototype")

    # Chat Tab
    with gr.Tab("Chat"):
        chat = gr.ChatInterface(
            fn=chat_handler,
            examples=["What's the weather?", "Debug this code snippet."],  # Placeholders
            title="AI Conversation (Mock Mode)"
        )

    # Config Tab
    with gr.Tab("Config"):
        config_text = gr.Textbox(
            value=yaml.dump(config),
            label="Edit Config (YAML)",
            lines=10
        )
        update_btn = gr.Button("Save Config")
        output = gr.Textbox(label="Status")
        update_btn.click(update_config, inputs=config_text, outputs=output)

    # Plugins Tab
    with gr.Tab("Plugins"):
        gr.Markdown("Manage custom skills here.")
        plugin_list = gr.Textbox(label="Current Plugins", interactive=False)
        refresh_btn = gr.Button("Refresh List")
        refresh_btn.click(list_plugins, outputs=plugin_list)
        gr.Markdown("To add: Place .py files in ./plugins/ with a 'run_skill()' function.")

    # Observation Tab
    with gr.Tab("Observation"):
        gr.Markdown(
            "**Privacy Notice:** Screen analysis runs locally only. "
            "No data is transmitted externally."
        )
        consent_check = gr.Checkbox(
            label="I consent to screen analysis for this session",
            value=False
        )
        observe_btn = gr.Button("Analyze Screen (Mock)")
        observe_output = gr.Textbox(label="Result")
        observe_btn.click(trigger_observation, inputs=consent_check, outputs=observe_output)

if __name__ == "__main__":
    app.launch(share=False, server_name="127.0.0.1")  # Local-only, no sharing