from transformers import BertModel, BertPreTrainedModel
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch
from .Layers import MultiLinearLayer

from transformers.adapters.mixins.bert import (
    BertModelWithHeadsAdaptersMixin,
)


class BertForConversationalQuestionAnswering(
    BertModelWithHeadsAdaptersMixin, BertPreTrainedModel
):
    def __init__(
        self,
        config,
        n_layers=2,
        activation="relu",
        beta=100,
    ):
        super(BertForConversationalQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        hidden_size = config.hidden_size
        self.rationale_l = MultiLinearLayer(
            n_layers, hidden_size, hidden_size, 1, activation
        )
        self.logits_l = MultiLinearLayer(
            n_layers, hidden_size, hidden_size, 2, activation
        )
        self.unk_l = MultiLinearLayer(n_layers, hidden_size, hidden_size, 1, activation)
        self.attention_l = MultiLinearLayer(
            n_layers, hidden_size, hidden_size, 1, activation
        )
        self.yn_l = MultiLinearLayer(n_layers, hidden_size, hidden_size, 2, activation)
        self.beta = beta

        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        rationale_mask=None,
        cls_idx=None,
        head_mask=None,
    ):

        outputs = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            return_dict=True,
        )

        # last layer hidden-states of sequence, first token:classification token
        final_hidden = outputs["last_hidden_state"]
        pooled_output = outputs["pooler_output"]
        # rationale_logits
        rationale_logits = self.rationale_l(final_hidden)
        rationale_logits = torch.sigmoid(rationale_logits)
        final_hidden = final_hidden * rationale_logits

        # attention layer to cal logits
        attention = self.attention_l(final_hidden).squeeze(-1)
        attention.data.masked_fill_(attention_mask.eq(0), -float("inf"))
        attention = F.softmax(attention, dim=-1)
        attention_pooled_output = (attention.unsqueeze(-1) * final_hidden).sum(dim=-2)

        # on to find answer in the article
        segment_mask = token_type_ids.type(final_hidden.dtype)
        rationale_logits = rationale_logits.squeeze(-1) * segment_mask

        # get span logits
        logits = self.logits_l(final_hidden)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        start_logits, end_logits = (
            start_logits * rationale_logits,
            end_logits * rationale_logits,
        )
        start_logits.data.masked_fill_(attention_mask.eq(0), -float("inf"))
        end_logits.data.masked_fill_(attention_mask.eq(0), -float("inf"))

        # cal unkown/yes/no logits
        unk_logits = self.unk_l(pooled_output)
        yn_logits = self.yn_l(attention_pooled_output)
        yes_logits, no_logits = yn_logits.split(1, dim=-1)

        # start_positions and end_positions is None when evaluate
        # return loss during training
        # return logits during evaluate
        if start_positions is not None and end_positions is not None:

            start_positions, end_positions = (
                start_positions + cls_idx,
                end_positions + cls_idx,
            )

            new_start_logits = torch.cat(
                (yes_logits, no_logits, unk_logits, start_logits), dim=-1
            )
            new_end_logits = torch.cat(
                (yes_logits, no_logits, unk_logits, end_logits), dim=-1
            )

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = new_start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            span_loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = span_loss_fct(new_start_logits, start_positions)
            end_loss = span_loss_fct(new_end_logits, end_positions)

            # rationale part
            alpha = 0.25
            gamma = 2.0

            # use rationale span to help calculate loss
            rationale_mask = rationale_mask.type(final_hidden.dtype)
            rationale_loss = -alpha * (
                (1 - rationale_logits) ** gamma
            ) * rationale_mask * torch.log(rationale_logits + 1e-7) - (1 - alpha) * (
                rationale_logits**gamma
            ) * (
                1 - rationale_mask
            ) * torch.log(
                1 - rationale_logits + 1e-7
            )

            rationale_loss = (rationale_loss * segment_mask).sum() / segment_mask.sum()

            total_loss = (start_loss + end_loss) / 2 + rationale_loss * self.beta
            return total_loss

        return start_logits, end_logits, yes_logits, no_logits, unk_logits
