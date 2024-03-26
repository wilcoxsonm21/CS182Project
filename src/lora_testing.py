from models import LoraTransformerModel

lora_config = LoraConfig(
                                    r=16,
                                    lora_alpha=16,
                                    target_modules=["attn.c_attn", "mlp.c_fc", "mlp.c_proj"],
                                    lora_dropout=0.0,
                                    bias="none")

model = LoraTransformerModel(n_dims=2, n_positions=256)

#print(model)

print("Trainable parameters:", model.get_trainable_params())
print("Non-trainable parameters:", model.get_non_trainable_params())