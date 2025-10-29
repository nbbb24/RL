#!/usr/bin/env python3
"""
Streamlit Chat Interface for ECG Expert Models
Based on chat.py - provides a web interface to chat with base, SFT, and GRPO models
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import json
from pathlib import Path
from datetime import datetime

# Page config
st.set_page_config(
    page_title="ECG Expert Chat",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitChatBot:
    def __init__(self, model_configs, system_prompt_file=None):
        """Initialize chatbot with multiple models"""
        self.models = {}
        self.tokenizers = {}
        self.system_prompt = self._load_system_prompt(system_prompt_file)
        
        # Load models
        for config in model_configs:
            self._load_model(config)
    
    def _load_system_prompt(self, prompt_file):
        """Load system prompt from file or use default"""
        if prompt_file and os.path.exists(prompt_file):
            try:
                with open(prompt_file, 'r') as f:
                    return f.read().strip()
            except:
                pass
        return "You are a helpful ECG expert assistant."
    
    def _load_model(self, config):
        """Load a single model"""
        name = config['name']
        model_path = config['model_path']
        adapter_path = config.get('adapter_path')
        
        try:
            with st.spinner(f"Loading {name} model..."):
                if adapter_path and os.path.exists(adapter_path):
                    # Load base model + LoRA adapters
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                    )
                    model = PeftModel.from_pretrained(model, adapter_path)
                    model.eval()
                else:
                    # Load full model directly
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                tokenizer.pad_token = tokenizer.eos_token
                
                self.models[name] = model
                self.tokenizers[name] = tokenizer
                
        except Exception as e:
            st.error(f"Failed to load {name} model: {str(e)}")
    
    def generate_response(self, model_name, user_input, conversation_history):
        """Generate response for a single model"""
        if model_name not in self.models:
            return f"Model {model_name} not loaded"
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Build conversation with system prompt
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_input})
        
        # Format prompt
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant reply
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            assistant_reply = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        else:
            assistant_reply = response.split(user_input)[-1].strip()
        
        return assistant_reply

def save_user_preference(question, responses, selected_model, timestamp):
    """Save user preference to JSON file"""
    try:
        # Use absolute path to ensure we're in the right directory
        current_dir = Path(__file__).parent
        hci_data_dir = current_dir / "data" / "hci_data"
        hci_data_dir.mkdir(parents=True, exist_ok=True)
        
        data_file = hci_data_dir / "user_chatbox_data.json"
        
        # Load existing data or create new
        if data_file.exists():
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading existing data: {e}")
                data = []
        else:
            data = []
        
        # Create new entry
        entry = {
            "timestamp": timestamp,
            "question": question,
            "responses": responses,
            "selected_model": selected_model,
            "total_models": len(responses)
        }
        
        data.append(entry)
        
        # Save back to file
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully saved preference to {data_file}")
        return True
        
    except Exception as e:
        print(f"Error saving preference: {e}")
        return False


def get_available_grpo_checkpoints():
    """Get available GRPO checkpoints"""
    grpo_base_path = Path("models/grpo/meta-llama/Llama-3.2-3B-Instruct")
    if not grpo_base_path.exists():
        return []
    
    checkpoints = []
    for checkpoint_dir in grpo_base_path.glob("global_step_*"):
        if checkpoint_dir.is_dir():
            step_num = checkpoint_dir.name.split("_")[-1]
            adapter_path = checkpoint_dir / "actor" / "lora_adapter"
            if adapter_path.exists():
                checkpoints.append(int(step_num))
    
    return sorted(checkpoints)

def main():
    st.title("🫀 ECG Expert Chat Interface")
    st.markdown("Chat with different versions of the ECG expert model")
    
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Model Configuration")
        
        # Model selection
        st.subheader("Select Models")
        base_model = st.checkbox("Base Model", value=True)
        sft_model = st.checkbox("SFT Model", value=False)
        grpo_model = st.checkbox("GRPO Model", value=False)
        
        # GRPO checkpoint selection
        if grpo_model:
            available_checkpoints = get_available_grpo_checkpoints()
            if available_checkpoints:
                grpo_checkpoint = st.selectbox(
                    "GRPO Checkpoint",
                    available_checkpoints,
                    index=len(available_checkpoints)-1  # Default to latest
                )
            else:
                st.warning("No GRPO checkpoints found")
                grpo_model = False
                grpo_checkpoint = 126
        else:
            grpo_checkpoint = 126
        
        # Advanced settings
        st.subheader("Advanced Settings")
        base_model_path = st.text_input(
            "Base Model Path",
            value="meta-llama/Llama-3.2-3B-Instruct"
        )
        sft_adapter_path = st.text_input(
            "SFT Adapter Path",
            value="models/sft"
        )
        system_prompt_file = st.text_input(
            "System Prompt File",
            value="data/system_prompt.txt"
        )
        gpu_device = st.text_input(
            "GPU Device",
            value="3",
            help="GPU device number (e.g., 0, 1, 2, 3)"
        )
        
        # Load models button
        load_models = st.button("🔄 Load Selected Models", type="primary")
    
    # Initialize session state
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "models_loaded" not in st.session_state:
        st.session_state.models_loaded = False
    if "last_responses" not in st.session_state:
        st.session_state.last_responses = {}
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""
    if "pending_responses" not in st.session_state:
        st.session_state.pending_responses = {}
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""
    if "show_vote_buttons" not in st.session_state:
        st.session_state.show_vote_buttons = False
    if "chosen_model" not in st.session_state:
        st.session_state.chosen_model = None
    
    # Load models when button is clicked
    if load_models:
        selected_models = []
        if base_model:
            selected_models.append("base")
        if sft_model:
            selected_models.append("sft")
        if grpo_model:
            selected_models.append("grpo")
        
        if not selected_models:
            st.error("Please select at least one model!")
        else:
            # Set GPU device
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device
            
            # Build model configs
            model_configs = []
            for model_name in selected_models:
                if model_name == 'base':
                    model_configs.append({
                        'name': 'base',
                        'model_path': base_model_path
                    })
                elif model_name == 'sft':
                    model_configs.append({
                        'name': 'sft',
                        'model_path': base_model_path,
                        'adapter_path': sft_adapter_path
                    })
                elif model_name == 'grpo':
                    grpo_adapter_path = f"models/grpo/meta-llama/Llama-3.2-3B-Instruct/global_step_{grpo_checkpoint}/actor/lora_adapter"
                    model_configs.append({
                        'name': f'grpo_step{grpo_checkpoint}',
                        'model_path': base_model_path,
                        'adapter_path': grpo_adapter_path
                    })
            
            # Load chatbot
            st.session_state.chatbot = StreamlitChatBot(model_configs, system_prompt_file)
            st.session_state.models_loaded = True
            st.success(f"✅ Loaded models: {', '.join(selected_models)}")
    
    # Chat interface
    if st.session_state.models_loaded and st.session_state.chatbot:
        st.header("💬 Chat")
        
        # Show loaded models info
        model_names = list(st.session_state.chatbot.models.keys())
        if len(model_names) == 1:
            st.info(f"🤖 **Loaded Model:** {model_names[0].upper()}")
        else:
            st.info(f"🤖 **Loaded Models:** {', '.join([name.upper() for name in model_names])} ({len(model_names)} models)")
        
        # Display conversation history (excluding the last question if it's still pending votes)
        history_to_display = st.session_state.conversation_history.copy()

        # If we have pending responses, don't show the last user message in history
        # (it will be shown in the voting section below)
        if st.session_state.last_responses and st.session_state.last_question:
            # Remove the last user message and assistant response from history display
            if len(history_to_display) >= 2 and history_to_display[-2]["role"] == "user":
                history_to_display = history_to_display[:-2]

        for i, message in enumerate(history_to_display):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                # For assistant messages, we'll show them in the same format as new responses
                if len(model_names) == 1:
                    with st.chat_message("assistant"):
                        st.write(message["content"])
                else:
                    # For multiple models, show the response in columns
                    cols = st.columns(len(model_names))
                    with cols[0]:  # Show the stored response in first column
                        with st.chat_message("assistant"):
                            st.write(message["content"])
        
        # Chat input
        st.markdown("---")
        user_input = st.chat_input("Ask about ECG or medical questions...")
        
        if user_input:
            # Add user message to history
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Generate responses from all loaded models
            model_names = list(st.session_state.chatbot.models.keys())
            
            # Generate responses from all models first
            responses = {}
            for model_name in model_names:
                with st.spinner(f"Generating response from {model_name}..."):
                    response = st.session_state.chatbot.generate_response(
                        model_name,
                        user_input,
                        st.session_state.conversation_history[:-1]
                    )
                    responses[model_name] = response

            # Store responses in session state BEFORE displaying
            st.session_state.last_responses = responses
            st.session_state.last_question = user_input

            # Don't add to conversation history yet - wait until user chooses best answer

        # Display the last responses if they exist (persists across button clicks)
        if st.session_state.last_responses and st.session_state.last_question:
            # Get model names from the stored responses, not from current loaded models
            response_model_names = list(st.session_state.last_responses.keys())

            # Display user question
            with st.chat_message("user"):
                st.write(st.session_state.last_question)

            # Display responses based on number of models
            if len(response_model_names) == 1:
                # Single model - full width
                with st.chat_message("assistant"):
                    model_name = response_model_names[0]
                    st.write(f"**{model_name.upper()}:**")
                    st.write(st.session_state.last_responses[model_name])

                    # Voting button for single model
                    if st.button(f"👍 Best Answer", key=f"vote_{model_name}"):
                        timestamp = datetime.now().isoformat()
                        save_user_preference(
                            st.session_state.last_question,
                            st.session_state.last_responses,
                            model_name,
                            timestamp
                        )
                        # Add the CHOSEN model's response to conversation history
                        st.session_state.conversation_history.append({
                            "role": "assistant",
                            "content": st.session_state.last_responses[model_name]
                        })
                        st.success(f"✅ Data saved! You chose {model_name.upper()} model!")
                        # Clear pending responses so question moves to history
                        st.session_state.last_responses = {}
                        st.session_state.last_question = ""
            else:
                # Multiple models - create columns
                cols = st.columns(len(response_model_names))

                for i, model_name in enumerate(response_model_names):
                    with cols[i]:
                        with st.chat_message("assistant"):
                            st.write(f"**{model_name.upper()}:**")
                            st.write(st.session_state.last_responses[model_name])

                            # Voting button for each model
                            if st.button(f"👍 Best Answer", key=f"vote_{model_name}"):
                                timestamp = datetime.now().isoformat()
                                save_user_preference(
                                    st.session_state.last_question,
                                    st.session_state.last_responses,
                                    model_name,
                                    timestamp
                                )
                                # Add the CHOSEN model's response to conversation history
                                st.session_state.conversation_history.append({
                                    "role": "assistant",
                                    "content": st.session_state.last_responses[model_name]
                                })
                                st.success(f"✅ Data saved! You chose {model_name.upper()}!")
                                # Clear pending responses so question moves to history
                                st.session_state.last_responses = {}
                                st.session_state.last_question = ""
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
    
    else:
        st.info("👆 Please configure and load models in the sidebar to start chatting!")
        
        # Show system prompt preview
        if os.path.exists(system_prompt_file):
            with st.expander("📋 System Prompt Preview"):
                with open(system_prompt_file, 'r') as f:
                    system_prompt = f.read().strip()
                st.text(system_prompt)
        
        # Show available GRPO checkpoints
        available_checkpoints = get_available_grpo_checkpoints()
        if available_checkpoints:
            with st.expander("📁 Available GRPO Checkpoints"):
                st.write(f"Found {len(available_checkpoints)} checkpoints:")
                for checkpoint in available_checkpoints:
                    st.write(f"- Step {checkpoint}")
        

if __name__ == "__main__":
    main()
