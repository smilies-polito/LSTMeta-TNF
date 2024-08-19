import torch
import numpy as np

class RNN(torch.nn.Module):
    def __init__(self,
        cell_states_dim, input_params_dim,
        n_layers_in=1,
        n_layers_out=1,
        n_layers_rnn=1,
        latent_dim=500,
        dropout_linear=0.2,
        dropout_rnn=0.,
        nonlinearity='relu',
    ):
        super().__init__()

        # ENCODER

        layers_dims = np.linspace(input_params_dim, latent_dim, n_layers_in+1, dtype=int)
        layers = []
        for idx in range(n_layers_in):
            layers += [
                torch.nn.Linear(layers_dims[idx], layers_dims[idx+1]),
                torch.nn.ReLU(),
            ]
            if idx < n_layers_in-1:
                layers.append(torch.nn.Dropout(dropout_linear))
            else:
                layers.append(torch.nn.Dropout(dropout_rnn))
        
        layers.append(torch.nn.Dropout(dropout_rnn))
        self.input_to_hidden = torch.nn.Sequential(*layers)

        # RNN

        self.n_layers_rnn = n_layers_rnn
        self.rnn = torch.nn.RNN(
            cell_states_dim,
            latent_dim,
            num_layers=n_layers_rnn,
            batch_first=True,
            dropout=dropout_rnn,
            nonlinearity=nonlinearity
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
    
    def forward(self, cell_states, input_params=None, hidden=None):
        """
            cell_states: (BATCH SIZE, LENGTH, CELL_STATES_DIM)
            input_params: (BATCH SIZE, INPUT_PARAMS_DIM)
            hidden: (N_LAYERS_RNN, BATCH SIZE, LATENT_DIM)
        """

        if hidden is None:
            assert input_params is not None, "Hidden state must be provided if not using input params"
            hidden = self.input_to_hidden(input_params)
            # hidden is now (BATCH SIZE, LATENT_DIM)
            hidden = hidden.repeat(self.n_layers_rnn, 1, 1)
        
        hidden_out, hidden_last = self.rnn(cell_states, hidden)

        cell_states_out = self.hidden_to_output(hidden_out)

        return cell_states_out, hidden_last
