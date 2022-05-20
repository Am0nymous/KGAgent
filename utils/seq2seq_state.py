
__all__ = [
    'State',
    "LSTMState",
    "TransformerState"
]

from typing import Union
import torch


class State:
    def __init__(self, encoder_output=None, encoder_mask=None, **kwargs):

        self.encoder_output = encoder_output
        self.encoder_mask = encoder_mask
        self._decode_length = 0

    @property
    def num_samples(self):

        if self.encoder_output is not None:
            return self.encoder_output.size(0)
        else:
            return None

    @property
    def decode_length(self):
        return self._decode_length

    @decode_length.setter
    def decode_length(self, value):
        self._decode_length = value

    def _reorder_state(self, state: Union[torch.Tensor, list, tuple], indices: torch.LongTensor, dim: int = 0):
        if isinstance(state, torch.Tensor):
            state = state.index_select(index=indices, dim=dim)
        elif isinstance(state, list):
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = self._reorder_state(state[i], indices, dim)
        elif isinstance(state, tuple):
            tmp_list = []
            for i in range(len(state)):
                assert state[i] is not None
                tmp_list.append(self._reorder_state(state[i], indices, dim))
            state = tuple(tmp_list)
        else:
            raise TypeError(f"Cannot reorder data of type:{type(state)}")

        return state

    def reorder_state(self, indices: torch.LongTensor):
        if self.encoder_mask is not None:
            self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        if self.encoder_output is not None:
            self.encoder_output = self._reorder_state(self.encoder_output, indices)


class LSTMState(State):
    def __init__(self, encoder_output, encoder_mask, hidden, cell):
        super().__init__(encoder_output, encoder_mask)
        self.hidden = hidden
        self.cell = cell
        self._input_feed = hidden[0]

    @property
    def input_feed(self):
        return self._input_feed

    @input_feed.setter
    def input_feed(self, value):
        self._input_feed = value

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.hidden = self._reorder_state(self.hidden, indices, dim=1)
        self.cell = self._reorder_state(self.cell, indices, dim=1)
        if self.input_feed is not None:
            self.input_feed = self._reorder_state(self.input_feed, indices, dim=0)


class TransformerState(State):
    def __init__(self, encoder_output, encoder_mask, num_decoder_layer):
        super().__init__(encoder_output, encoder_mask)
        self.encoder_key = [None] * num_decoder_layer
        self.encoder_value = [None] * num_decoder_layer
        self.decoder_prev_key = [None] * num_decoder_layer
        self.decoder_prev_value = [None] * num_decoder_layer

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.encoder_key = self._reorder_state(self.encoder_key, indices)
        self.encoder_value = self._reorder_state(self.encoder_value, indices)
        self.decoder_prev_key = self._reorder_state(self.decoder_prev_key, indices)
        self.decoder_prev_value = self._reorder_state(self.decoder_prev_value, indices)

    @property
    def decode_length(self):
        if self.decoder_prev_key[0] is not None:
            return self.decoder_prev_key[0].size(1)
        return 0


