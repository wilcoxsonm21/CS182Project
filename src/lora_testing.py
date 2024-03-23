from models import LoraTransformerModel

model = LoraTransformerModel(n_dims=2, n_positions=256)

#print(model)

print("Trainable parameters:", model.get_trainable_params())
print("Non-trainable parameters:", model.get_non_trainable_params())