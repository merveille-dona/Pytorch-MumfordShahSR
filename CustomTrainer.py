import pathlib
import torch
import torch.nn
import torch.nn.utils
import torch.optim
import torch.optim.lr_scheduler

import ignite.engine
import numpy
import typing

class CustomEngine(ignite.engine.Engine):

    def __init__(self, 
        process_function: typing.Callable[[ignite.engine.Engine, typing.Any], typing.Any]
    ):

        super(CustomEngine, self).__init__(process_function)

        # Add attribute for compute epoch loss
        self.epoch_loss = 0.
        self.counter = 0

        # Epoch loss history
        self.epoch_loss_history = []

def create_train_step(
    model: torch.nn.Module,
    model_device: torch.device,
    datas_device: torch.device,
    optimizer: torch.optim.Optimizer, 
    criterion
):

    # Define any training logic for iteration update
    def train_step(engine, batch):
        
        inputs, results = batch[0], batch[1]
        
        # Move batch on model_device    
        # inputs = inputs.to(model_device, non_blocking=True)
        # results = results.to(model_device, non_blocking=True)

        model.train()
        
        optimizer.zero_grad()
        
        # Batch processing
        batch_loss = 0
        size_of_batch = len(inputs)
        predictions = []
        for i in range(0, size_of_batch):

            res_size = results[i].size()
            inp_size = inputs[i].size()
            decim_row = res_size[0] // inp_size[0]
            decim_col = res_size[1] // inp_size[1]

            # to model device
            x = inputs[i].to(model_device, non_blocking=True)
            y = results[i].to(model_device, non_blocking=True)

            prediction = model(x, decim_row, decim_col)
            predictions.append(prediction.unsqueeze(0).to(datas_device))
            loss: torch.Tensor = criterion(prediction, y)
            batch_loss += loss.item()
            loss.backward()

            # return on datas device
            inputs[i] = x.to(datas_device, non_blocking=True)
            results[i] = y.to(datas_device, non_blocking=True)

        optimizer.step()

        batch_loss /= size_of_batch

        # Move prediction on datas device for Evaluator
        # predictions = torch.concat(predictions, axis=0)
        # predictions = predictions.to(datas_device, non_blocking=True)
        
        # Batch return on datas device
        # inputs = inputs.to(datas_device, non_blocking=True)
        # results = results.to(datas_device, non_blocking=True)
        
        output = {
            'prediction' : predictions,
            'result' : results,
            'batch_loss' : batch_loss
        }

        
        return output

    return train_step



def update_epoch_loss(engine: CustomEngine) -> None:
    # Update batch loss
    bl = engine.state.output['batch_loss']
    engine.epoch_loss += bl
    # Update counter
    engine.counter += 1
    
def compute_epoch_loss(engine: CustomEngine) -> None:
    # Epoch loss is mean of all batch loss
    engine.epoch_loss /= engine.counter
    engine.epoch_loss_history.append(engine.epoch_loss)
    # Reset loss
    engine.epoch_loss = 0.
    engine.counter = 0

def save_epoch_loss(engine: CustomEngine, path: pathlib.Path) -> None:
    numpy.save(
        file = path,
        arr = numpy.array(engine.epoch_loss_history)
    )

def print_logs(engine: CustomEngine):
    strp = 'Epoch [{}/{}] : Loss {:.6f}'
    epoch_loss = engine.epoch_loss_history[engine.state.epoch]
    print(
        strp.format(
            engine.state.epoch,
            engine.state.max_epochs,
            epoch_loss
        )
    )

def save_model(
    engine: ignite.engine.Engine, 
    model: torch.nn.Module, 
    path: pathlib.Path = pathlib.Path('.')
) -> None:
    no_epoch = engine.state.epoch
    torch.save(model.state_dict(), path / 'model_epoch_{}.pt'.format(no_epoch))