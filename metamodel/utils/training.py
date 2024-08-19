import torch
import numpy as np


class EarlyStopping():
    """
        Early stopping callback that stops training when validation loss stops improving.

        Parameters:
        -----------
        tolerance : int
            Number of epochs to wait before stopping if no improvement is seen.
        min_delta : float
            Minimum change in the monitored quantity to qualify as an improvement.
    """

    def __init__(self, lr_tolerance, es_tolerance, min_delta):

        self.lr_tolerance = lr_tolerance
        self.es_tolerance = es_tolerance
        self.min_delta = min_delta

        self.reduced_lr = False
        self.early_stop = False

    def __call__(self, val_loss_list):

        if len(val_loss_list) < max(self.lr_tolerance, self.es_tolerance):
            return
        
        val_loss_tocheck = (
            np.array(val_loss_list[-self.es_tolerance:]) if self.reduced_lr
            else np.array(val_loss_list[-self.lr_tolerance:])
        )

        # Take care of increasing loss or outliers
        indices = np.where(val_loss_tocheck[1:] > val_loss_tocheck[:-1])[0]
        max_n_idx = (
            int(0.55*self.es_tolerance) if self.reduced_lr
            else int(0.55*self.lr_tolerance)
        )
        if len(indices) >= max_n_idx:
            if self.reduced_lr : self.early_stop = True
            self.reduced_lr = True
            return

        # Take care of plateauing loss
        if np.allclose(
            val_loss_tocheck[:-1], # all but the last element
            val_loss_tocheck[1:], # all but the first element
            rtol=0, # relative tolerance
            atol=self.min_delta # absolute tolerance
        ):
            if self.reduced_lr : self.early_stop = True
            self.reduced_lr = True


def calculate_hidden_l2norm(hidden):
    return hidden.norm(2, dim=1)

def update_gradients_norm(model, grads_norm_dict):
    for name, p in model.named_parameters():
        name = "gradients/" + name
        if p.grad is not None:
            gradients = p.grad.detach()
            if name in grads_norm_dict.keys():
                grads_norm_dict[name] += gradients.norm(2).item()
            else:
                grads_norm_dict[name] = gradients.norm(2).item()
    return grads_norm_dict

def update_weights_norm(model, weights_norm_dict):
    for name, p in model.named_parameters():
        if "weight" in name:
            name = "weights/" + name
            weight = p.detach()
            if name in weights_norm_dict.keys():
                weights_norm_dict[name] += weight.norm(2).item()
            else:
                weights_norm_dict[name] = weight.norm(2).item()
    return weights_norm_dict

def update_hidden_norm(hidden, hidden_norm_dict, name='hidden_norm'):
    name = "hidden_layer/" + name
    hidden_norm = calculate_hidden_l2norm(hidden).mean()
    if name in hidden_norm_dict.keys():
        hidden_norm_dict[name] += hidden_norm.detach().item()
    else:
        hidden_norm_dict[name] = hidden_norm.detach().item()
    return hidden_norm_dict

def update_losses(losses, running_loss_dict, names):
    for name, loss in zip(names, losses):
        name = "loss/" + name
        if name in running_loss_dict.keys():
            running_loss_dict[name] += loss.detach().item()
        else:
            running_loss_dict[name] = loss.detach().item()
    return running_loss_dict


def train_epoch_bptt(
    model, optimizer, loss_fn, device, dataloader, n_batches, cell_states_norm_factor=None, clip_value=None,
):
    
    running_loss = [None]*n_batches
    hidden_states = [None]*n_batches

    for window_current, data in enumerate(dataloader):
        input_params_batches, _, cell_states_batches, labels_batches = data
        
        batch_current = 0
        # Since the data is windowed, we need to iterate over batches now
        for input_params, cell_states, labels in zip(
            input_params_batches.squeeze(dim=0),
            cell_states_batches.squeeze(dim=0),
            labels_batches.squeeze(dim=0)
        ):

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            cell_states = cell_states.to(device)
            labels = labels.to(device)

            # Make predictions for this batch
            if window_current == 0:
                input_params = input_params.to(device)
                cell_states_out, hidden_last = model(
                    cell_states, input_params=input_params
                )
            else:
                cell_states_out, hidden_last = model(
                    cell_states,
                    hidden=hidden_states[batch_current]
                )

            # Compute the loss and its gradients
            if cell_states_norm_factor:
                loss = torch.mean(torch.sum(loss_fn(
                    cell_states_out * torch.Tensor(cell_states_norm_factor).to(device),
                    labels * torch.Tensor(cell_states_norm_factor).to(device),
                    axis=-1
                )))
            else:
                loss = torch.mean(torch.sum(loss_fn(cell_states_out, labels), axis=-1))
            loss.backward()

            if running_loss[batch_current] is None:
                running_loss[batch_current] = [loss.detach().item()]
            else:
                running_loss[batch_current].append(loss.detach().item())

            # Clip gradients
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Adjust learning weights
            optimizer.step()

            if type(hidden_last) == tuple:
                hidden_states[batch_current] = tuple([h.detach() for h in hidden_last])
            else:
                hidden_states[batch_current] = hidden_last.detach()
            batch_current += 1
        
    avg_loss = np.mean([np.mean(x) for x in running_loss])
    return avg_loss


def test_epoch_bptt(
    model, loss_fn, device, dataloader, n_batches
):
    
    input_params_trend = [None]*n_batches
    init_cell_states_trend = [None]*n_batches
    cell_states_trend = [None]*n_batches
    cell_states_pred = [None]*n_batches
    running_vloss = [None]*n_batches
    hidden_states = [None]*n_batches
    
    with torch.no_grad():
        
        for window_current, vdata in enumerate(dataloader):
            input_params_batches, init_cell_states_batches, cell_states_batches, labels_batches = vdata
            
            batch_current = 0
            # Since the data is windowed, we need to iterate over batches now
            for input_params, init_cell_states, cell_states, labels in zip(
                input_params_batches.squeeze(dim=0),
                init_cell_states_batches.squeeze(dim=0),
                cell_states_batches.squeeze(dim=0),
                labels_batches.squeeze(dim=0)
            ):

                cell_states = cell_states.to(device)
                labels = labels.to(device)
                
                # Make predictions for this batch
                if window_current == 0:
                    input_params_trend[batch_current] = input_params.detach().numpy()
                    init_cell_states_trend[batch_current] = init_cell_states.detach().numpy()
                    input_params = input_params.to(device)
                    cell_states_out, hidden_last = model(
                        cell_states, input_params=input_params
                    )
                else:
                    cell_states_out, hidden_last = model(
                        cell_states_out,
                        hidden=hidden_states[batch_current]
                    )

                # Compute the loss
                vloss = torch.mean(torch.sum(loss_fn(cell_states_out, labels), axis=-1))

                if window_current == 0:
                    running_vloss[batch_current] = [vloss.detach().item()]
                    cell_states_trend[batch_current] =\
                        np.expand_dims(cell_states.detach().cpu().numpy()[:,0,:], 1)
                    cell_states_pred[batch_current] =\
                        np.expand_dims(cell_states.detach().cpu().numpy()[:,0,:], 1)
                else:
                    running_vloss[batch_current].append(vloss.detach().item())

                cell_states_trend[batch_current] = np.concatenate(
                    [cell_states_trend[batch_current], labels.detach().cpu().numpy()],
                    axis=1
                )
                cell_states_pred[batch_current] = np.concatenate(
                    [cell_states_pred[batch_current], cell_states_out.detach().cpu().numpy()],
                    axis=1
                )

                if type(hidden_last) == tuple:
                    hidden_states[batch_current] = tuple([h.detach() for h in hidden_last])
                else:
                    hidden_states[batch_current] = hidden_last.detach()
                batch_current += 1

    avg_vloss = np.mean([np.mean(x) for x in running_vloss])
    input_params_trend = np.stack(input_params_trend, axis=0)
    init_cell_states_trend = np.stack(init_cell_states_trend, axis=0)
    cell_states_trend = np.stack(cell_states_trend, axis=0)
    cell_states_pred = np.stack(cell_states_pred, axis=0)
        
    return avg_vloss, input_params_trend, init_cell_states_trend, cell_states_trend, cell_states_pred


def train_epoch_bptt_tnfvector(
    model, optimizer, loss_fn, device, dataloader, n_batches, cell_states_norm_factor=None, clip_value=None,
):
    
    running_loss = [None]*n_batches
    hidden_states = [None]*n_batches

    for window_current, data in enumerate(dataloader):
        input_params_batches, _, cell_states_batches, tnf_vector_batches, labels_batches = data
        
        batch_current = 0
        # Since the data is windowed, we need to iterate over batches now
        for input_params, cell_states, tnf_vector, labels in zip(
            input_params_batches.squeeze(dim=0),
            cell_states_batches.squeeze(dim=0),
            tnf_vector_batches.squeeze(dim=0),
            labels_batches.squeeze(dim=0)
        ):

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            cell_states = cell_states.to(device)
            tnf_vector = tnf_vector.to(device)
            labels = labels.to(device)

            # Make predictions for this batch
            if window_current == 0:
                input_params = input_params.to(device)
                cell_states_out, hidden_last = model(
                    cell_states, tnf_vector=tnf_vector, input_params=input_params
                )
            else:
                cell_states_out, hidden_last = model(
                    cell_states,
                    tnf_vector=tnf_vector,
                    hidden=hidden_states[batch_current]
                )

            # Compute the loss and its gradients
            if cell_states_norm_factor:
                loss = torch.mean(torch.sum(loss_fn(
                    cell_states_out * torch.Tensor(cell_states_norm_factor).to(device),
                    labels * torch.Tensor(cell_states_norm_factor).to(device),
                    axis=-1
                )))
            else:
                loss = torch.mean(torch.sum(loss_fn(cell_states_out, labels), axis=-1))
            loss.backward()

            if running_loss[batch_current] is None:
                running_loss[batch_current] = [loss.detach().item()]
            else:
                running_loss[batch_current].append(loss.detach().item())

            # Clip gradients
            if clip_value is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            # Adjust learning weights
            optimizer.step()

            if type(hidden_last) == tuple:
                hidden_states[batch_current] = tuple([h.detach() for h in hidden_last])
            else:
                hidden_states[batch_current] = hidden_last.detach()
            batch_current += 1
        
    avg_loss = np.mean([np.mean(x) for x in running_loss])
    return avg_loss


def test_epoch_bptt_tnfvector(
    model, loss_fn, device, dataloader, n_batches
):
    
    input_params_trend = [None]*n_batches
    init_cell_states_trend = [None]*n_batches
    cell_states_trend = [None]*n_batches
    cell_states_pred = [None]*n_batches
    running_vloss = [None]*n_batches
    hidden_states = [None]*n_batches
    
    with torch.no_grad():
        
        for window_current, vdata in enumerate(dataloader):
            input_params_batches, init_cell_states_batches, cell_states_batches, tnf_vector_batches, labels_batches = vdata
            
            batch_current = 0
            # Since the data is windowed, we need to iterate over batches now
            for input_params, init_cell_states, cell_states, tnf_vector, labels in zip(
                input_params_batches.squeeze(dim=0),
                init_cell_states_batches.squeeze(dim=0),
                cell_states_batches.squeeze(dim=0),
                tnf_vector_batches.squeeze(dim=0),
                labels_batches.squeeze(dim=0)
            ):

                cell_states = cell_states.to(device)
                tnf_vector = tnf_vector.to(device)
                labels = labels.to(device)
                
                # Make predictions for this batch
                if window_current == 0:
                    input_params_trend[batch_current] = input_params.detach().numpy()
                    init_cell_states_trend[batch_current] = init_cell_states.detach().numpy()
                    input_params = input_params.to(device)
                    cell_states_out, hidden_last = model(
                        cell_states, tnf_vector=tnf_vector, input_params=input_params
                    )
                else:
                    cell_states_out, hidden_last = model(
                        cell_states_out,
                        tnf_vector=tnf_vector,
                        hidden=hidden_states[batch_current]
                    )

                # Compute the loss
                vloss = torch.mean(torch.sum(loss_fn(cell_states_out, labels), axis=-1))

                if window_current == 0:
                    running_vloss[batch_current] = [vloss.detach().item()]
                    cell_states_trend[batch_current] =\
                        np.expand_dims(cell_states.detach().cpu().numpy()[:,0,:], 1)
                    cell_states_pred[batch_current] =\
                        np.expand_dims(cell_states.detach().cpu().numpy()[:,0,:], 1)
                else:
                    running_vloss[batch_current].append(vloss.detach().item())

                cell_states_trend[batch_current] = np.concatenate(
                    [cell_states_trend[batch_current], labels.detach().cpu().numpy()],
                    axis=1
                )
                cell_states_pred[batch_current] = np.concatenate(
                    [cell_states_pred[batch_current], cell_states_out.detach().cpu().numpy()],
                    axis=1
                )

                if type(hidden_last) == tuple:
                    hidden_states[batch_current] = tuple([h.detach() for h in hidden_last])
                else:
                    hidden_states[batch_current] = hidden_last.detach()
                batch_current += 1

    avg_vloss = np.mean([np.mean(x) for x in running_vloss])
    input_params_trend = np.stack(input_params_trend, axis=0)
    init_cell_states_trend = np.stack(init_cell_states_trend, axis=0)
    cell_states_trend = np.stack(cell_states_trend, axis=0)
    cell_states_pred = np.stack(cell_states_pred, axis=0)
        
    return avg_vloss, input_params_trend, init_cell_states_trend, cell_states_trend, cell_states_pred
