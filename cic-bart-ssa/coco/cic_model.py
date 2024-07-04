import torch

from modeling_bart import VLBart
class VLBartPretraining(VLBart):
    def __init__(self, config):
        super().__init__(config)

        self.losses = self.config.losses.split(',')

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            return_dict=True,
            reduce_loss=reduce_loss
        )
        assert 'loss' in output

        loss = output['loss']

        result = {
            'loss': loss
        }
        return result

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        lm_labels = batch["target_ids"].to(device)

        reduce_loss = True
        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            labels=lm_labels,
            return_dict=True,
            reduce_loss=reduce_loss
        )
        assert 'loss' in output

        loss = output['loss']

        results = {
            'loss': loss
        }
       
        if 'ground_caption' in self.losses:
            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                max_length=100,
                num_beams=5,
            )

            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

            results['ground_caption'] = generated_sents

        return results

    @torch.no_grad()
    def generate_step(self, batch):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        vis_attention_mask = None
        if 'vis_attention_mask' in batch:
            vis_attention_mask = batch['vis_attention_mask'].to(device)

        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            vis_attention_mask=vis_attention_mask,
            max_length=100,
            num_beams=5,
        )

        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)

        return generated_sents