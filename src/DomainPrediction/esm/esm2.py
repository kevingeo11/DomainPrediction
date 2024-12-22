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

    def get_res(self, sequence, rep_layer=33):
        data = [
            ("protein1", sequence)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.device == 'gpu':
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[rep_layer], return_contacts=True)

        return results

    def get_res_batch(self, sequences, rep_layer=33):
        data = [
            (f"P{i+1}", seq) for i, seq in enumerate(sequences)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.device == 'gpu':
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[rep_layer], return_contacts=True)

        return results, batch_lens

    def get_logits(self, sequence):

        results = self.get_res(sequence)
        return results['logits']

    def get_prob(self, sequence):
        logits = self.get_logits(sequence)
        prob = torch.nn.functional.softmax(logits, dim=-1)[0, 1:-1, :] # 1st and last are start and end tokens

        return prob.cpu().numpy()
    

class ESM2_iterate():
    def __init__(self, model_path, device='cpu') -> None:
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.device = device

        if self.device == 'gpu':
            self.model.cuda()

        self.tok_to_idx = self.alphabet.tok_to_idx
        self.idx_to_tok = {v:k for k,v in self.tok_to_idx.items()}

    def random_fill(self, sequence, infill_idx, fix=2, verbose=False):
        masked_query = ''.join(['<mask>' if i in infill_idx else sequence[i] for i in range(len(sequence))])

        data = [
            ("protein1", masked_query)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.device == 'gpu':
            batch_tokens = batch_tokens.cuda()

        ## Random Mask Approach
        with torch.no_grad():
            for i in range(len(infill_idx)):
                if verbose:
                    print(f'run {i}')
                
                results =self. model(batch_tokens, repr_layers=[33], return_contacts=True)
                prob = torch.nn.functional.softmax(results['logits'], dim=-1)
                masked_positions = torch.nonzero(batch_tokens == 32, as_tuple=False)
                
                if verbose:
                    print(f'num maxed pos {masked_positions.size(0)}')
                
                if masked_positions.size(0) > fix:
                    selected_positions = masked_positions[torch.randperm(masked_positions.size(0))[:fix]]
                    batch_tokens[0, selected_positions[:, 1]] = torch.multinomial(prob[:, selected_positions[:, 1], :][0], 1).flatten()
                else:
                    selected_positions = masked_positions
                    batch_tokens[0, selected_positions[:, 1]] = torch.multinomial(prob[:, selected_positions[:, 1], :][0], 1).flatten()

                if verbose:
                    print(f" T domain : {''.join([self.idx_to_tok[i] if self.idx_to_tok[i] != '<mask>' else '_' for i in batch_tokens[0, 1:-1][infill_idx].numpy()])}")

                if torch.nonzero(batch_tokens == 32, as_tuple=False).size(0) == 0:
                    break

        new_sequence = ''.join([self.idx_to_tok[i] for i in batch_tokens.cpu().numpy()[0][1:-1]])

        assert len(sequence) == len(new_sequence)
        assert ''.join([sequence[i] for i in range(len(sequence)) if i not in infill_idx]) == ''.join([new_sequence[i] for i in range(len(new_sequence)) if i not in infill_idx])

        return new_sequence
    
    def entropy_fill(self, sequence, infill_idx, fix=2, verbose=False):
        masked_query = ''.join(['<mask>' if i in infill_idx else sequence[i] for i in range(len(sequence))])

        data = [
            ("protein1", masked_query)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.device == 'gpu':
            batch_tokens = batch_tokens.cuda()

        ## Min Entropy position
        with torch.no_grad():
            for i in range(len(infill_idx)):
                if verbose:
                    print(f'run {i}')

                results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
                prob = torch.nn.functional.softmax(results['logits'], dim=-1)
                masked_positions = torch.nonzero(batch_tokens == 32, as_tuple=False)

                if verbose:
                    print(f'num maxed pos {masked_positions.size(0)}')
                
                if masked_positions.size(0) > fix:
                    masked_probs = prob[0 ,masked_positions[:, 1], :]
                    entropies = -torch.sum(masked_probs * torch.log(masked_probs + 1e-10), dim=-1)
                    indices = torch.argsort(entropies)[:fix]
                    selected_positions = masked_positions[:, 1][indices]
                    batch_tokens[0, selected_positions] = torch.multinomial(prob[:, selected_positions, :][0], 1).flatten()
                else:
                    selected_positions = masked_positions[:, 1]
                    batch_tokens[0, selected_positions] = torch.multinomial(prob[:, selected_positions, :][0], 1).flatten()
                
                if verbose:
                    print(f" T domain : {''.join([self.idx_to_tok[i] if self.idx_to_tok[i] != '<mask>' else '_' for i in batch_tokens[0, 1:-1][infill_idx].numpy()])}")

                if torch.nonzero(batch_tokens == 32, as_tuple=False).size(0) == 0:
                    break

        new_sequence = ''.join([self.idx_to_tok[i] for i in batch_tokens.cpu().numpy()[0][1:-1]])

        assert len(sequence) == len(new_sequence)
        assert ''.join([sequence[i] for i in range(len(sequence)) if i not in infill_idx]) == ''.join([new_sequence[i] for i in range(len(new_sequence)) if i not in infill_idx])

        return new_sequence