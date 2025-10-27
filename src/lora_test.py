from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading base model '{model_name}'...")

base_model=AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,)

print("Loading LoRA adapter...")

model=PeftModel.from_pretrained(base_model, "./lora_adapter")
tokenizer=AutoTokenizer.from_pretrained("./lora_adapter")
def generate_answer(question,context):
    prompt = f"""Question: {question}
    Contexte: {context}
    Réponse:"""
    inputs=tokenizer(prompt,return_tensors="pt").to(model.device)
    outputs=model.generate(**inputs,max_new_tokens=256
                           ,do_smple=True,
                           temperature=0.7,
                           pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0],skip_special_tokens=True)

question = "Qu'est-ce que le deep learning?"
context = "[Document 1] Le deep learning utilise des réseaux de neurones profonds."

response = generate_answer(question, context)
print(response)