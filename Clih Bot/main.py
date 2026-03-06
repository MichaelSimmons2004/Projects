from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import httpx
import yaml

import ai_engine as ai_bridge
from ai_engine import get_ai_response_stream
from observation import analyze_screen
from storage import load_config, save_config

BASE_DIR = Path(__file__).parent
API_BASE_URL = os.getenv("CLIHBOT_API_BASE_URL", "http://localhost:8765")
DEFAULT_SERVER_URL = "http://localhost:1234/v1"

def _ensure_backend_api() -> None:
    try:
        ai_bridge._ensure_backend()
    except Exception:
        pass



def api_request(endpoint: str, method: str = "GET", data: dict | None = None, timeout: float = 10.0) -> dict:
    url = f"{API_BASE_URL}{endpoint}"
    for attempt in range(2):
        try:
            with httpx.Client(timeout=timeout) as client:
                if method == "POST":
                    response = client.post(url, json=data or {})
                elif method == "DELETE":
                    response = client.delete(url)
                else:
                    response = client.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            if attempt == 0:
                _ensure_backend_api()
                continue
            return {"error": str(exc)}
    return {"error": "Unknown API failure"}



def get_runtime_state() -> dict:
    config = api_request("/model/config")
    if "error" in config:
        return {
            "ready": False,
            "notice": f"Unable to load runtime configuration: {config['error']}",
            "show_launch": False,
            "config": None,
            "status": None,
            "config_error": config["error"],
        }

    allow_auto_recover = bool(
        config.get("model_auto_load", True) and config.get("lmstudio_auto_start", True)
    )
    health = api_request("/health")
    status = api_request("/model/status")

    if "error" in health or "error" in status:
        if allow_auto_recover:
            return {"ready": True, "notice": "", "show_launch": False, "config": config, "status": {}, "config_error": None}
        return {
            "ready": False,
            "notice": "No live CLIHBot backend is available. Review preferences or launch LM Studio.",
            "show_launch": True,
            "config": config,
            "status": {},
            "config_error": None,
        }

    if health.get("lmstudio_reachable") or health.get("model_loaded") or health.get("loaded_models"):
        return {"ready": True, "notice": "", "show_launch": False, "config": config, "status": status, "config_error": None}

    if allow_auto_recover:
        return {"ready": True, "notice": "", "show_launch": False, "config": config, "status": status, "config_error": None}

    current_url = status.get("current_llm_url") or status.get("server_url") or config.get("lmstudio_base_url", DEFAULT_SERVER_URL)
    return {
        "ready": False,
        "notice": f"No live model server is currently detected at `{current_url}`. Review preferences or launch LM Studio.",
        "show_launch": True,
        "config": config,
        "status": status,
        "config_error": None,
    }



def build_runtime_summary() -> str:
    state = get_runtime_state()
    if state.get("config_error"):
        return "\n".join([
            "**Backend** `configuration unavailable`",
            f"**Error** `{state['config_error']}`",
        ])

    status = state.get("status", {})
    if not status:
        return "\n".join([
            "**Backend** `unavailable`",
            "**Model** `unknown`",
            f"**Server** `{state['config'].get('lmstudio_base_url', DEFAULT_SERVER_URL)}`",
        ])

    current_url = status.get("current_llm_url") or status.get("server_url") or "unknown"
    loaded_models = status.get("models", [])
    active_model = loaded_models[0] if loaded_models else "none"
    external = status.get("external_server_detected", False)
    return "\n".join([
        f"**Server** `{current_url}`",
        f"**Model** `{active_model}`",
        f"**External Preferred** `{status.get('prefer_external', False)}`",
        f"**External Detected** `{external}`",
    ])


def build_selector_runtime_summary(bootstrap: dict) -> str:
    status = bootstrap.get("status", {}) if isinstance(bootstrap, dict) else {}
    gpu = bootstrap.get("gpu", {}) if isinstance(bootstrap, dict) else {}

    current_url = status.get("server_url") or "unknown"
    loaded_models = status.get("models", []) or []
    active_model = loaded_models[0] if loaded_models else "none"
    gpu_name = gpu.get("gpu_name") or "GPU"
    free_mb = gpu.get("free_mb")

    lines = [
        f"**Server** `{current_url}`",
        f"**Model** `{active_model}`",
        f"**External Preferred** `{status.get('prefer_external', False)}`",
        f"**External Detected** `{status.get('external_server_detected', False)}`",
    ]
    if free_mb is not None:
        lines.append(f"**GPU Free VRAM** `{free_mb} MB` on `{gpu_name}`")
    if status.get("error"):
        lines.append(f"**Status Error** `{status['error']}`")
    return "\n".join(lines)


def render_ranked_models(models: list[dict]) -> str:
    if not models:
        return "No LM Studio models available for ranking."

    lines = [
        "| Score | Model | Params | Weights | Est VRAM | Context | Quant | Notes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for model in models[:12]:
        breakdown = model.get("score_breakdown", {}).get("components", {})
        top_notes = sorted(
            (
                comp.get("reason", "")
                for comp in breakdown.values()
                if comp.get("contribution", 0) > 0
            ),
            key=len,
        )[:2]
        notes = "; ".join(top_notes) if top_notes else "No positive scoring signals"
        lines.append(
            f"| {model.get('rank_score', 0):.1f} | `{model.get('id', 'unknown')}` | "
            f"{model.get('parameter_count_b', model.get('size_gb', 'n/a'))}B | "
            f"{model.get('file_size_gb', 'n/a')} GB | "
            f"{model.get('estimated_vram_gb', 'n/a')} GB | "
            f"{model.get('context_window', 'n/a')} | "
            f"{model.get('quantization', 'n/a')} | {notes} |"
        )
    return "\n".join(lines)


def build_manual_preference_choices(models: list[dict]) -> list[str]:
    model_ids: list[str] = []
    families: list[str] = []
    for model in models or []:
        model_id = model.get("id")
        family = model.get("family")
        if model_id and model_id not in model_ids:
            model_ids.append(model_id)
        if family and family not in families:
            families.append(family)
    return families + model_ids


def build_selector_message(bootstrap: dict, base_message: str = "") -> str:
    lines = [base_message] if base_message else []
    gpu = bootstrap.get("gpu", {}) if isinstance(bootstrap, dict) else {}
    selected = bootstrap.get("selected_model") if isinstance(bootstrap, dict) else None
    selected_estimate = bootstrap.get("selected_model_estimate") if isinstance(bootstrap, dict) else None

    gpu_name = gpu.get("gpu_name") or "GPU"
    total_mb = gpu.get("total_mb") or 0
    free_mb = gpu.get("free_mb") or 0
    if total_mb > 0:
        lines.append(f"Detected `{gpu_name}` with `{free_mb}` MB free of `{total_mb}` MB total.")
    elif gpu_name:
        lines.append(f"Detected `{gpu_name}`, but VRAM totals are unavailable.")

    if selected and selected.get("id"):
        lines.append(
            f"Current top candidate: `{selected['id']}` "
            f"(score `{selected.get('rank_score', 0):.1f}`, weights `{selected.get('file_size_gb', 'n/a')} GB`, "
            f"custom-runtime heuristic VRAM `{selected.get('estimated_vram_gb', 'n/a')} GB`)."
        )
    if selected_estimate and not selected_estimate.get("error"):
        lines.append(
            f"LM Studio default-load estimate at `{selected_estimate.get('context_length', 'n/a')}` tokens: "
            f"`{selected_estimate.get('estimated_gpu_memory_gib', 'n/a')} GiB` GPU, "
            f"`{selected_estimate.get('estimated_total_memory_gib', 'n/a')} GiB` total."
        )
        if selected_estimate.get("note"):
            lines.append(selected_estimate["note"])
    elif selected_estimate and selected_estimate.get("error"):
        lines.append(f"LM Studio estimate unavailable: `{selected_estimate['error']}`")

    return "\n\n".join(line for line in lines if line)



def load_preferences():
    config = api_request("/model/config")
    status = api_request("/model/status")
    if "error" in config:
        error_message = f"Unable to load runtime configuration: {config['error']}"
        return (
            False,
            False,
            False,
            False,
            "",
            build_runtime_summary(),
            error_message,
        )
    if "error" in status:
        status = {}
    return (
        status.get("auto_detect_external", config.get("auto_detect_external_server", True)),
        status.get("prefer_external", config.get("prefer_external_servers", True)),
        config.get("model_auto_load", True),
        config.get("lmstudio_auto_start", True),
        config.get("lmstudio_base_url", DEFAULT_SERVER_URL),
        build_runtime_summary(),
        "",
    )



def apply_preferences(auto_detect: bool, prefer_external: bool, model_auto_load: bool, lmstudio_auto_start: bool, server_url: str):
    config = api_request("/model/config")
    if "error" in config:
        return build_runtime_summary(), f"Preference update blocked: {config['error']}"

    messages = []

    updates = {
        "auto_detect_external_server": auto_detect,
        "prefer_external_servers": prefer_external,
        "model_auto_load": model_auto_load,
        "lmstudio_auto_start": lmstudio_auto_start,
    }
    result = api_request("/model/config/update", method="POST", data=updates)
    if "error" in result:
        messages.append(f"Preference update failed: {result['error']}")
    else:
        messages.append(result.get("message", "Preferences updated."))

    if server_url:
        url_result = api_request("/model/set-url", method="POST", data={"base_url": server_url})
        if "error" in url_result:
            messages.append(f"Server URL update failed: {url_result['error']}")
        else:
            messages.append(url_result.get("message", "Server URL updated."))

    return build_runtime_summary(), "\n".join(messages)



def load_model_selection_settings():
    bootstrap = api_request("/model/selector/bootstrap")
    if "error" in bootstrap:
        return (
            "auto",
            True,
            None,
            150000,
            25.0,
            20.0,
            15.0,
            15.0,
            10.0,
            10.0,
            10.0,
            "Selector bootstrap unavailable.",
            f"Unable to load model selection settings: {bootstrap['error']}",
            "Unable to load rankings.",
            gr.update(choices=[]),
        )

    config = bootstrap.get("config", {})
    ranking = bootstrap.get("ranking", {})
    gpu = bootstrap.get("gpu", {})
    raw_weights = ranking.get("raw_weights", {}) if "error" not in ranking else {}
    configured_budget = config.get("max_vram_mb")
    live_budget = None
    gpu_message = ""
    if "error" in gpu:
        gpu_message = f"Live GPU measurements unavailable: {gpu['error']}"
    else:
        total_mb = gpu.get("total_mb") or 0
        free_mb = gpu.get("free_mb") or 0
        gpu_name = gpu.get("gpu_name") or "GPU"
        live_budget = free_mb if free_mb > 0 else None
        if total_mb > 0:
            gpu_message = f"Detected `{gpu_name}` with `{free_mb}` MB free of `{total_mb}` MB total."
        else:
            gpu_message = f"Detected `{gpu_name}`, but VRAM totals are unavailable."
    return (
        ranking.get("mode", config.get("ranking_mode", "auto")),
        config.get("prefer_larger_context", True),
        configured_budget if configured_budget not in (None, "") else live_budget,
        config.get("context_target_tokens", config.get("min_context_tokens", 150000)),
        raw_weights.get("family", config.get("model_family_weight", 25.0)),
        raw_weights.get("context", config.get("context_window_weight", 20.0)),
        raw_weights.get("quant", config.get("quantization_weight", 15.0)),
        raw_weights.get("size", config.get("size_weight", 15.0)),
        raw_weights.get("loaded", config.get("loaded_weight", 10.0)),
        raw_weights.get("vram", config.get("vram_weight", 10.0)),
        raw_weights.get("manual", config.get("manual_weight", 10.0)),
        build_selector_runtime_summary(bootstrap),
        build_selector_message(bootstrap, gpu_message),
        render_ranked_models(bootstrap.get("ranked_models", [])),
        gr.update(choices=build_manual_preference_choices(bootstrap.get("ranked_models", []))),
    )


def build_ranked_models_markdown(manual_preference: str) -> str:
    payload = {"manual_preference": manual_preference} if manual_preference else {}
    result = api_request("/model/selector/available", method="POST", data=payload, timeout=30.0)
    if "error" in result:
        return f"Unable to load ranked models: {result['error']}"
    return render_ranked_models(result.get("models", []))


def refresh_ranked_models(manual_preference: str):
    bootstrap = api_request("/model/selector/bootstrap")
    summary = build_selector_runtime_summary(bootstrap) if "error" not in bootstrap else "Selector bootstrap unavailable."
    message = build_selector_message(bootstrap, "Ranked models refreshed.") if "error" not in bootstrap else "Ranked models refreshed."
    return (
        summary,
        message,
        build_ranked_models_markdown(manual_preference or "") if manual_preference else render_ranked_models(bootstrap.get("ranked_models", [])),
        gr.update(choices=build_manual_preference_choices(bootstrap.get("ranked_models", []))),
    )


def use_live_gpu_budget():
    gpu = api_request("/model/gpu-memory")
    if "error" in gpu:
        return gr.update(), f"Live GPU measurements unavailable: {gpu['error']}"

    total_mb = gpu.get("total_mb") or 0
    free_mb = gpu.get("free_mb") or 0
    gpu_name = gpu.get("gpu_name") or "GPU"
    if free_mb <= 0:
        return gr.update(), f"Detected `{gpu_name}`, but free VRAM could not be determined."

    message = f"Set VRAM budget to live free memory on `{gpu_name}`: `{free_mb}` MB"
    if total_mb > 0:
        message += f" of `{total_mb}` MB total."
    return gr.update(value=free_mb), message


def save_model_selection_settings(
    ranking_mode: str,
    prefer_larger_context: bool,
    max_vram_mb,
    min_context_tokens,
    family_weight: float,
    context_weight: float,
    quant_weight: float,
    size_weight: float,
    loaded_weight: float,
    vram_weight: float,
    manual_weight: float,
    manual_preference: str,
):
    payload = {
        "mode": ranking_mode,
        "prefer_larger_context": prefer_larger_context,
        "max_vram_mb": int(max_vram_mb) if max_vram_mb not in (None, "") else None,
        "min_context_tokens": int(min_context_tokens or 0),
        "family_weight": float(family_weight),
        "context_weight": float(context_weight),
        "quant_weight": float(quant_weight),
        "size_weight": float(size_weight),
        "loaded_weight": float(loaded_weight),
        "vram_weight": float(vram_weight),
        "manual_weight": float(manual_weight),
    }
    result = api_request("/model/selector/ranking-config", method="POST", data=payload, timeout=30.0)
    if "error" in result:
        message = f"Model selection update failed: {result['error']}"
    else:
        message = result.get("message", "Model selection settings updated.")
    bootstrap = api_request("/model/selector/bootstrap")
    summary = build_selector_runtime_summary(bootstrap) if "error" not in bootstrap else "Selector bootstrap unavailable."
    display_message = build_selector_message(bootstrap, message) if "error" not in bootstrap else message
    ranked = build_ranked_models_markdown(manual_preference or "")
    choices = build_manual_preference_choices(bootstrap.get("ranked_models", [])) if "error" not in bootstrap else []
    return summary, display_message, ranked, gr.update(choices=choices)


def preview_best_model(manual_preference: str):
    payload = {"manual_preference": manual_preference} if manual_preference else {}
    result = api_request("/model/selector/best", method="POST", data=payload, timeout=30.0)
    if "error" in result:
        message = f"Preview failed: {result['error']}"
    else:
        selected = result.get("selected_model") or {}
        message = result.get("reason", "No ranking result.")
        if selected.get("id"):
            message = f"{message}\n\nSelected candidate: `{selected['id']}`"
    bootstrap = api_request("/model/selector/bootstrap")
    summary = build_selector_runtime_summary(bootstrap) if "error" not in bootstrap else "Selector bootstrap unavailable."
    display_message = build_selector_message(bootstrap, message) if "error" not in bootstrap else message
    ranked = build_ranked_models_markdown(manual_preference or "")
    choices = build_manual_preference_choices(bootstrap.get("ranked_models", [])) if "error" not in bootstrap else []
    return summary, display_message, ranked, gr.update(choices=choices)


def select_and_load_best_model(manual_preference: str):
    payload = {"manual_preference": manual_preference} if manual_preference else {}
    result = api_request("/model/selector/select-and-load", method="POST", data=payload, timeout=60.0)
    if "error" in result:
        message = f"Select-and-load failed: {result['error']}"
    else:
        selected = result.get("selected_model") or {}
        message = result.get("reason", "Model selection attempted.")
        if selected.get("id"):
            message = f"{message}\n\nLoaded candidate: `{selected['id']}`"
    bootstrap = api_request("/model/selector/bootstrap")
    summary = build_selector_runtime_summary(bootstrap) if "error" not in bootstrap else "Selector bootstrap unavailable."
    display_message = build_selector_message(bootstrap, message) if "error" not in bootstrap else message
    ranked = build_ranked_models_markdown(manual_preference or "")
    choices = build_manual_preference_choices(bootstrap.get("ranked_models", [])) if "error" not in bootstrap else []
    return summary, display_message, ranked, gr.update(choices=choices)


def clear_manual_preference():
    return gr.update(value=None), "Manual preference cleared. Auto ranking will apply normally."



def launch_lmstudio_from_preferences():
    result = api_request("/model/start-lmstudio", method="POST", data={}, timeout=30.0)
    if "error" in result:
        message = f"Launch failed: {result['error']}"
    else:
        message = result.get("message", "LM Studio launch requested.")
    return build_runtime_summary(), message



def launch_lmstudio_from_chat():
    result = api_request("/model/start-lmstudio", method="POST", data={}, timeout=30.0)
    if "error" in result:
        notice = f"Launch failed: {result['error']}"
        show = True
    else:
        notice = result.get("message", "LM Studio launch requested.")
        show = False
    return gr.update(value=notice, visible=bool(notice)), gr.update(visible=show)



def chat_handler(message: str, history):
    if not message or not message.strip():
        yield "", gr.update(visible=False), gr.update(visible=False)
        return

    state = get_runtime_state()
    if not state["ready"]:
        yield "", gr.update(value=state["notice"], visible=True), gr.update(visible=state["show_launch"])
        return

    accumulated = ""
    for chunk in get_ai_response_stream(message):
        accumulated += chunk
        yield accumulated, gr.update(visible=False), gr.update(visible=False)

    if accumulated.startswith("[CLIHBot startup error") or accumulated.startswith("[Backend error"):
        yield accumulated, gr.update(value=accumulated, visible=True), gr.update(visible=True)



def update_config(new_yaml: str) -> str:
    try:
        new_config = yaml.safe_load(new_yaml) or {}
        save_config(new_config)
        return "Config updated successfully."
    except Exception as exc:
        return f"Error: {exc}"



def trigger_observation(user_consent: bool) -> str:
    if not user_consent:
        return "Consent required."
    return analyze_screen()



def list_plugins() -> str:
    plugins_dir = BASE_DIR / "plugins"
    if not plugins_dir.exists():
        return "No plugins found."
    plugins = sorted(f for f in os.listdir(plugins_dir) if f.endswith(".py"))
    return "\n".join(plugins) if plugins else "No plugins found."



def get_model_list_display():
    result = api_request("/model/list")
    if "error" in result:
        return "Unable to load models.", gr.update(choices=[], value=None)

    models = result.get("models", [])
    if not models:
        return "No models reported.", gr.update(choices=[], value=None)

    choices = []
    lines = []
    selected = None
    for model in models:
        model_id = model.get("id") if isinstance(model, dict) else None
        if not model_id:
            continue
        choices.append(model_id)
        if selected is None or model.get("is_loaded"):
            selected = model_id
        lines.append(f"- `{model_id}`")

    if not lines:
        return "No models reported.", gr.update(choices=[], value=None)
    return "\n".join(lines), gr.update(choices=choices, value=selected)



def select_model(model_id: str):
    if not model_id:
        return build_runtime_summary(), "Select a model first."

    result = api_request("/model/select", method="POST", data={"model_id": model_id}, timeout=30.0)
    if "error" in result:
        message = f"Selection failed: {result['error']}"
    elif result.get("success"):
        message = f"Selected `{model_id}`."
    else:
        message = result.get("reason", "Model selection failed.")
    return build_runtime_summary(), message


with gr.Blocks(title="LocalAI Copilot") as app:
    gr.Markdown("# CLIHBot - Local AI Assistant")

    with gr.Tab("Chat"):
        chat_notice = gr.Markdown(visible=False)
        launch_lmstudio_chat_btn = gr.Button("Launch LM Studio", visible=False)
        chatbot = gr.Chatbot(
            layout="bubble",
            label="ClihBot",
            height="75vh",
            autoscroll=True,
            elem_id="chatbot",
            group_consecutive_messages=True,
        )
        textbox = gr.Textbox(
            placeholder="Type your message here...",
            scale=2,
        )
        chat = gr.ChatInterface(
            fn=chat_handler,
            chatbot=chatbot,
            textbox=textbox,
            additional_outputs=[chat_notice, launch_lmstudio_chat_btn],
            title="Conversation",
            fill_height=True,
            fill_width=True,
            submit_btn=True,
            description="Talk to your local AI assistant",
            examples=[
                "What's the weather?",
                "Debug this code snippet.",
                "Summarize this article.",
                "What plugins do I have?",
            ],
        )

    with gr.Tab("Preferences"):
        gr.Markdown("### Runtime")
        preferences_status = gr.Markdown("Loading runtime status...")
        preference_message = gr.Markdown()
        with gr.Group():
            gr.Markdown("**Connection**")
            gr.Markdown("Choose where CLIHBot should look for a live model server.")
            server_url_input = gr.Textbox(label="Preferred Server URL")
            auto_detect_toggle = gr.Checkbox(label="Detect running external servers")
            prefer_external_toggle = gr.Checkbox(label="Use a detected external server when available")

        with gr.Group():
            gr.Markdown("**Automation**")
            gr.Markdown("These settings control what happens when you send a message and nothing is loaded yet.")
            model_auto_load_toggle = gr.Checkbox(label="Load a model automatically when needed")
            lmstudio_auto_start_toggle = gr.Checkbox(label="Start LM Studio automatically if no server is available")

        with gr.Row():
            apply_preferences_btn = gr.Button("Save Preferences", variant="primary")
            refresh_preferences_btn = gr.Button("Refresh Runtime")
            launch_lmstudio_pref_btn = gr.Button("Launch LM Studio Now")

        app.load(
            load_preferences,
            outputs=[
                auto_detect_toggle,
                prefer_external_toggle,
                model_auto_load_toggle,
                lmstudio_auto_start_toggle,
                server_url_input,
                preferences_status,
                preference_message,
            ],
        )

    with gr.Tab("Model Selection"):
        gr.Markdown("### Model Selection")
        selector_status = gr.Markdown("Loading model selector...")
        selector_message = gr.Markdown()
        manual_preference_input = gr.Dropdown(
            label="Manual Preference",
            info="Optional. Choose a family or exact model. Leave empty for normal auto selection.",
            choices=[],
            allow_custom_value=True,
            value=None,
        )
        with gr.Row():
            ranking_mode_dropdown = gr.Dropdown(
                label="Ranking Mode",
                choices=["auto", "manual"],
                value="auto",
            )
            prefer_larger_context_selector = gr.Checkbox(label="Prefer larger context windows")
        with gr.Row():
            max_vram_mb_input = gr.Number(label="Max VRAM Budget (MB)", precision=0)
            min_context_tokens_input = gr.Number(label="Recommended Context Target", precision=0, value=150000)
        with gr.Row():
            detect_vram_btn = gr.Button("Use Live Free VRAM")
            refresh_rankings_btn = gr.Button("Refresh Ranked Models")
            clear_manual_pref_btn = gr.Button("Clear Manual Preference")
        with gr.Accordion("Ranking Weights", open=False):
            family_weight_slider = gr.Slider(0, 100, value=25, step=1, label="Family Match")
            context_weight_slider = gr.Slider(0, 100, value=20, step=1, label="Context Window")
            quant_weight_slider = gr.Slider(0, 100, value=15, step=1, label="Quantization Quality")
            size_weight_slider = gr.Slider(0, 100, value=15, step=1, label="Model Efficiency")
            loaded_weight_slider = gr.Slider(0, 100, value=10, step=1, label="Already Loaded")
            vram_weight_slider = gr.Slider(0, 100, value=10, step=1, label="VRAM Fit")
            manual_weight_slider = gr.Slider(0, 100, value=10, step=1, label="Manual Preference")
        with gr.Row():
            save_selector_btn = gr.Button("Save Model Rules", variant="primary")
            preview_best_btn = gr.Button("Preview Best Match")
            select_and_load_btn = gr.Button("Select and Load Best")
        ranked_models_output = gr.Markdown("Loading ranked models...")

        app.load(
            load_model_selection_settings,
            outputs=[
                ranking_mode_dropdown,
                prefer_larger_context_selector,
                max_vram_mb_input,
                min_context_tokens_input,
                family_weight_slider,
                context_weight_slider,
                quant_weight_slider,
                size_weight_slider,
                loaded_weight_slider,
                vram_weight_slider,
                manual_weight_slider,
                selector_status,
                selector_message,
                ranked_models_output,
                manual_preference_input,
            ],
        )

    with gr.Tab("Config"):
        config_text = gr.Textbox(
            value=yaml.dump(load_config()),
            label="Edit Config (YAML)",
            lines=10,
        )
        update_btn = gr.Button("Save Config")
        config_output = gr.Textbox(label="Status")
        update_btn.click(update_config, inputs=config_text, outputs=config_output)

    with gr.Tab("Plugins"):
        gr.Markdown("Manage custom skills here.")
        plugin_list = gr.Textbox(label="Current Plugins", interactive=False)
        refresh_plugins_btn = gr.Button("Refresh List")
        refresh_plugins_btn.click(list_plugins, outputs=plugin_list)
        gr.Markdown("To add: Place .py files in ./plugins/ with a 'run_skill()' function.")

    with gr.Tab("Observation"):
        gr.Markdown(
            "**Privacy Notice:** Screen analysis runs locally only. No data is transmitted externally."
        )
        consent_check = gr.Checkbox(label="I consent to screen analysis for this session", value=False)
        observe_btn = gr.Button("Analyze Screen")
        observe_output = gr.Textbox(label="Result")
        observe_btn.click(trigger_observation, inputs=consent_check, outputs=observe_output)

    launch_lmstudio_chat_btn.click(
        launch_lmstudio_from_chat,
        outputs=[chat_notice, launch_lmstudio_chat_btn],
    )

    apply_preferences_btn.click(
        apply_preferences,
        inputs=[
            auto_detect_toggle,
            prefer_external_toggle,
            model_auto_load_toggle,
            lmstudio_auto_start_toggle,
            server_url_input,
        ],
        outputs=[preferences_status, preference_message],
    )
    refresh_preferences_btn.click(
        lambda: (build_runtime_summary(), ""),
        outputs=[preferences_status, preference_message],
    )
    launch_lmstudio_pref_btn.click(
        launch_lmstudio_from_preferences,
        outputs=[preferences_status, preference_message],
    )
    save_selector_btn.click(
        save_model_selection_settings,
        inputs=[
            ranking_mode_dropdown,
            prefer_larger_context_selector,
            max_vram_mb_input,
            min_context_tokens_input,
            family_weight_slider,
            context_weight_slider,
            quant_weight_slider,
            size_weight_slider,
            loaded_weight_slider,
            vram_weight_slider,
            manual_weight_slider,
            manual_preference_input,
        ],
        outputs=[selector_status, selector_message, ranked_models_output, manual_preference_input],
    )
    detect_vram_btn.click(
        use_live_gpu_budget,
        outputs=[max_vram_mb_input, selector_message],
    )
    refresh_rankings_btn.click(
        refresh_ranked_models,
        inputs=manual_preference_input,
        outputs=[selector_status, selector_message, ranked_models_output, manual_preference_input],
    )
    preview_best_btn.click(
        preview_best_model,
        inputs=manual_preference_input,
        outputs=[selector_status, selector_message, ranked_models_output, manual_preference_input],
    )
    select_and_load_btn.click(
        select_and_load_best_model,
        inputs=manual_preference_input,
        outputs=[selector_status, selector_message, ranked_models_output, manual_preference_input],
    )
    clear_manual_pref_btn.click(
        clear_manual_preference,
        outputs=[manual_preference_input, selector_message],
    )


if __name__ == "__main__":
    app.launch(share=False, server_name="127.0.0.1")


