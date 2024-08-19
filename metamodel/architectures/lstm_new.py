import torch
import numpy as np

class LSTM(torch.nn.Module):
    def __init__(self,
        cell_states_dim, input_params_dim,
        n_layers_in=1,
        n_layers_out=1,
        n_layers_lstm=1,
        latent_dim=500,
        dropout_linear=0.2,
        dropout_lstm=0.,
        tnf_vector=False
    ):
        super().__init__()

        # ENCODER

        layers_dims = np.linspace(input_params_dim, latent_dim, n_layers_in+1, dtype=int)
        layers = []
        for idx in range(n_layers_in-1):
            layers += [
                torch.nn.Linear(layers_dims[idx], layers_dims[idx+1]),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_linear)
            ]
        
        input_to_hidden = layers + [
            torch.nn.Linear(layers_dims[-2], layers_dims[-1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_lstm)
        ]
        input_to_lstmcell = layers + [
            torch.nn.Linear(layers_dims[-2], layers_dims[-1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_lstm)
        ]
        self.input_to_hidden = torch.nn.Sequential(*input_to_hidden)
        self.input_to_lstmcell = torch.nn.Sequential(*input_to_lstmcell)

        self.cell_states_to_hidden = torch.nn.Sequential(
            torch.nn.Linear(cell_states_dim, latent_dim//2),
            torch.nn.ReLU()
        )
        if tnf_vector:
            self.tnf_vector_to_hidden = torch.nn.Sequential(
                torch.nn.Linear(1, latent_dim//2),
                torch.nn.ReLU()
            )

        # LSTM

        self.n_layers_lstm = n_layers_lstm
        self.lstm = torch.nn.LSTM(
            latent_dim//2, #cell_states_dim+1 if tnf_vector else cell_states_dim,
            latent_dim,
            num_layers=n_layers_lstm,
            batch_first=True,
            dropout=dropout_lstm,
        )

        # DECODER

        layers_dims = np.linspace(latent_dim, cell_states_dim, n_layers_out+1, dtype=int)
        layers = []
        for idx in range(n_layers_out):
            layers += [
                torch.nn.Linear(layers_dims[idx], layers_dims[idx+1])
            ]
            if idx < n_layers_out-1:
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout_linear))
            else:
                layers.append(torch.nn.Sigmoid())

        self.hidden_to_output = torch.nn.Sequential(*layers)

    def decode(self, x):
        decoded_lin = self.decoder_lin(x)
        
        return decoded_lin
    
    def forward(self, cell_states, tnf_vector=None, input_params=None, hidden=None):
        """
            cell_states: (BATCH SIZE, LENGTH, CELL_STATES_DIM)
            input_params: (BATCH SIZE, INPUT_PARAMS_DIM)
            hidden: tuple of 2 (N_LAYERS_RNN, BATCH SIZE, LATENT_DIM)
        """

        if hidden is None:
            if input_params is None:
                raise ValueError(
                    "Hidden state must be provided if not using input params"
                )
            hidden = self.input_to_hidden(input_params)
            lstmcell = self.input_to_lstmcell(input_params)
            # hidden and lstmcell are (BATCH SIZE, LATENT_DIM)
            hidden = hidden.repeat(self.n_layers_lstm, 1, 1)
            lstmcell = lstmcell.repeat(self.n_layers_lstm, 1, 1)
            hidden = (hidden, lstmcell)
        
        if tnf_vector is not None:
            inputs = (
                self.cell_states_to_hidden(cell_states) +
                self.tnf_vector_to_hidden(tnf_vector)
            )
        else:
            inputs = cell_states
        hidden_out, hidden_last = self.lstm(inputs, hidden)

        cell_states_out = self.hidden_to_output(hidden_out)

        return cell_states_out, hidden_last
