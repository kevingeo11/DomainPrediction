import torch
import esm


class ESM2():
    def __init__(self, model_path, device='cpu') -> None:
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.device = device

        if self.device == 'gpu':
            self.model.cuda()

        self.tok_to_idx = self.alphabet.tok_to_idx
        self.idx_to_tok = {v:k for k,v in self.tok_to_idx.items()}

    def get_res(self, sequence):
        data = [
            ("protein1", sequence)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.device == 'gpu':
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)

        return results

    def get_logits(self, sequence):

        results = self.get_res(sequence)
        return results['logits']

    def get_prob(self, sequence):
        logits = self.get_logits(sequence)
        prob = torch.nn.functional.softmax(logits, dim=-1)[0, 1:-1, :] # 1st and last are start and end tokens

        return prob.cpu().numpy()