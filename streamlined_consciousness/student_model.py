"""
Student Model (Local Learner)
Manages the local HuggingFace model and its PEFT/LoRA adapter.
"""

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from .config import config

# Configure logging
logger = logging.getLogger("student-model")

class StudentModel:
    def __init__(self):
        self.model_id = config.STUDENT_MODEL_ID
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # MPS support for Mac M-series
        if torch.backends.mps.is_available():
            self.device = "mps"
            
        self.model = None
        self.tokenizer = None
        self.adapter_path = "adapters/active_mind"
        
        # Ensure adapter directory exists
        os.makedirs(os.path.dirname(self.adapter_path), exist_ok=True)

    def load(self):
        """Load the model and adapter in 4-bit quantization"""
        logger.info(f"Loading Student Model: {self.model_id} on {self.device}")
        
        try:
            # Quantization config for memory efficiency
            # Note: BitsAndBytes might have issues on MPS (Mac), fallback to float16 if needed
            if self.device == "mps":
                logger.info("MPS detected: Using float16 (BitsAndBytes 4-bit not supported on MPS)")
                quantization_config = None
                torch_dtype = torch.float16
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                torch_dtype = None

            # Load Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load Model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device != "mps" else None, # MPS often requires manual move
                trust_remote_code=True
            )
            
            if self.device == "mps":
                self.model.to(self.device)

            # Initialize or Load LoRA Adapter
            if os.path.exists(self.adapter_path):
                logger.info(f"Loading existing adapter from {self.adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, self.adapter_path, is_trainable=True)
            else:
                logger.info("Initializing new LoRA adapter")
                peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False, 
                    r=config.LORA_RANK,
                    lora_alpha=config.LORA_ALPHA,
                    lora_dropout=0.05,
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"] # Common for Llama/Gemma
                )
                self.model = get_peft_model(self.model, peft_config)
            
            # Get trainable parameters info
            trainable_params, all_param = self.model.get_nb_trainable_parameters()
            logger.info(
                f"trainable params: {trainable_params:,} || "
                f"all params: {all_param:,} || "
                f"trainable%: {100 * trainable_params / all_param:.4f}"
            )
            logger.info("Student Model Loaded Successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Student Model: {e}")
            raise

    def generate(self, prompt: str, max_new_tokens: int = 100) -> str:
        """Generate response (Shadow Mode)"""
        if not self.model:
            raise RuntimeError("Model not loaded")
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def save_adapter(self):
        """Save the current adapter state"""
        if self.model:
            self.model.save_pretrained(self.adapter_path)
            logger.info(f"Adapter saved to {self.adapter_path}")

    def get_hidden_size(self):
        """Robustly get the hidden size of the model"""
        if not self.model:
            return 3072 # Default fallback
            
        config = self.model.config
        
        # Try common attribute names
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        elif hasattr(config, "d_model"):
            return config.d_model
        elif hasattr(config, "dim"):
            return config.dim
        
        # Try to infer from embedding layer
        try:
            return self.model.get_input_embeddings().weight.shape[1]
        except:
            pass
            
        logger.warning("Could not determine hidden size, defaulting to 3072")
        return 3072
