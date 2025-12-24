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
        self.project_id = "default"
        self.adapter_path = self.get_adapter_path(self.project_id)
        
        # Ensure adapter directory exists
        os.makedirs(os.path.dirname(self.adapter_path), exist_ok=True)

    def get_adapter_path(self, project_id: str) -> str:
        """Get the adapter path for a specific project"""
        # Sanitize model ID for path usage (replace / with _)
        model_slug = self.model_id.replace("/", "_")
        return os.path.join(config.ADAPTERS_ROOT_DIR, project_id, model_slug)

    def switch_project(self, project_id: str):
        """Switch the active adapter to a different project"""
        if project_id == self.project_id:
            return
            
        logger.info(f"Switching project from {self.project_id} to {project_id}")
        
        # Save current state if model is loaded
        if self.model:
            self.save_adapter()
            
        self.project_id = project_id
        self.adapter_path = self.get_adapter_path(project_id)
        os.makedirs(os.path.dirname(self.adapter_path), exist_ok=True)
        
        # If model is loaded, we need to reload the adapter
        if self.model:
            try:
                # Get the underlying base model
                # PeftModel -> Base Model
                if hasattr(self.model, "unload"):
                    base_model = self.model.unload()
                else:
                    # Fallback if unload not available (older PEFT)
                    base_model = self.model.base_model
                
                # Re-initialize adapter
                if os.path.exists(self.adapter_path):
                    logger.info(f"Loading existing adapter from {self.adapter_path}")
                    self.model = PeftModel.from_pretrained(base_model, self.adapter_path, is_trainable=True)
                else:
                    logger.info(f"Initializing new adapter for {project_id}")
                    peft_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False, 
                        r=config.LORA_RANK,
                        lora_alpha=config.LORA_ALPHA,
                        lora_dropout=0.05,
                        target_modules=config.LORA_TARGET_MODULES
                    )
                    self.model = get_peft_model(base_model, peft_config)
                    
                logger.info(f"Switched to project: {project_id}")
                
            except Exception as e:
                logger.error(f"Failed to switch project adapter: {e}")
                # Try to recover by reloading full model
                self.load()

    def load(self):
        """Load the model and adapter in 4-bit quantization"""
        logger.info(f"Loading Student Model: {self.model_id} on {self.device}")
        
        try:
            # Quantization config for memory efficiency
            # Note: BitsAndBytes might have issues on MPS (Mac), fallback to float32 for stability
            if self.device == "mps":
                logger.info("MPS detected: Using float32 (BitsAndBytes 4-bit not supported on MPS)")
                quantization_config = None
                torch_dtype = torch.float32
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
                    target_modules=config.LORA_TARGET_MODULES
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
            
        cfg = self.model.config
        
        # Priority list of attribute names
        attributes = ["hidden_size", "d_model", "dim", "n_embd", "embedding_size"]
        
        for attr in attributes:
            if hasattr(cfg, attr):
                val = getattr(cfg, attr)
                if isinstance(val, int) and val > 0:
                    logger.info(f"Detected hidden size from config.{attr}: {val}")
                    return val
        
        # Try to infer from embedding layer
        try:
            val = self.model.get_input_embeddings().weight.shape[1]
            logger.info(f"Detected hidden size from embedding layer: {val}")
            return val
        except:
            pass
            
        logger.warning("Could not determine hidden size, defaulting to 3072")
        return 3072
