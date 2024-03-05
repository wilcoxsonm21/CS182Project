import torch
from eval import get_soft_model_from_run, get_model_from_run
from models import TransformerModel

class TransformerModelSoftPrompt(TransformerModel):

    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, n_soft_prompts=8):
        super(TransformerModelSoftPrompt, self).__init__(n_dims, n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head)

        self.n_soft_prompts = n_soft_prompts
        self.soft_prompts = torch.nn.Parameter(torch.randn(n_soft_prompts, n_embd), requires_grad=True)

    def freeze_except_soft_prompts(self):

        for param in self.parameters():
            param.requires_grad = False

        self.soft_prompts.requires_grad = True

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)

        soft_prompt_embeddings = self.soft_prompts.unsqueeze(0).expand(xs.shape[0], self.n_soft_prompts, -1)
        embeds = torch.cat((soft_prompt_embeddings, self._read_in(zs)), dim=1) 
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs

# I made the get_soft_model_from_run
model, conf = get_soft_model_from_run("../models/kernel_linear_regression/model1", step=500000, 
                                      model_class=TransformerModelSoftPrompt, n_soft_prompts=8)
model.freeze_except_soft_prompts()

x = torch.randn(2, 5, 1)
y = torch.randn(2, 5, 1)

print(model(x, y))