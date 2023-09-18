import pathlib
import torch
import torch.nn
import torch.nn.utils
import torch.optim
import torch.optim.lr_scheduler

import ignite.engine
import numpy

def create_train_step(
    model: torch.nn.Module,
    model_device: torch.device,
    datas_device: torch.device,
    optimizer: torch.optim.Optimizer, 
    criterion,
    lr_scheduler: torch.optim.lr_scheduler.StepLR = None
):

    # Define any training logic for iteration update
    def train_step(engine, batch):
        
        inputs, results = batch[0], batch[1]
        
        # Move batch on model_device    
        inputs = inputs.to(model_device, non_blocking=True)
        results = results.to(model_device, non_blocking=True)

        model.train()
        
        optimizer.zero_grad()
        

        # Batch processing
        batch_loss = 0
        size_of_batch = inputs.shape[0]
        predictions = []
        for i in range(0, size_of_batch):
            res_size = results[i].size()
            inp_size = inputs[i].size()
            decim_row = res_size[0] // inp_size[0]
            decim_col = res_size[1] // inp_size[1]
            prediction = model(inputs[i], decim_row, decim_col)
            predictions.append(prediction.unsqueeze(0))
            loss: torch.Tensor = criterion(prediction, results[i])
            batch_loss += loss.item()
            loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        optimizer.step()

        batch_loss /= size_of_batch
        
        
        
        if not(lr_scheduler is None):
            lr_scheduler.step()

        # Move prediction on datas device for Evaluator
        predictions = torch.concat(predictions, axis=0)
        predictions = predictions.to(datas_device, non_blocking=True)
        
        # Batch return on datas device
        inputs = inputs.to(datas_device, non_blocking=True)
        results = results.to(datas_device, non_blocking=True)
        
        output = {
            'prediction' : predictions,
            'result' : results,
            'loss' : batch_loss
        }

        
        return output

    return train_step



def create_train_step_with_variable_size_image(
    model: torch.nn.Module,
    model_device: torch.device,
    datas_device: torch.device,
    optimizer: torch.optim.Optimizer, 
    criterion,
    lr_scheduler: torch.optim.lr_scheduler.StepLR = None
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
        
        
        
        if not(lr_scheduler is None):
            lr_scheduler.step()

        # Move prediction on datas device for Evaluator
        # predictions = torch.concat(predictions, axis=0)
        # predictions = predictions.to(datas_device, non_blocking=True)
        
        # Batch return on datas device
        # inputs = inputs.to(datas_device, non_blocking=True)
        # results = results.to(datas_device, non_blocking=True)
        
        output = {
            'prediction' : predictions,
            'result' : results,
            'loss' : batch_loss
        }

        
        return output

    return train_step



def create_train_step_with_hidden_loss(
    model: torch.nn.Module,
    model_device: torch.device,
    datas_device: torch.device,
    optimizer: torch.optim.Optimizer, 
    criterion,
    lr_scheduler: torch.optim.lr_scheduler.StepLR = None
):

    # Define any training logic for iteration update
    def train_step(engine, batch):
        
        inputs, results = batch[0], batch[1]
        
        # Move batch on model_device    
        inputs = inputs.to(model_device, non_blocking=True)
        results = results.to(model_device, non_blocking=True)

        model.train()
        
        optimizer.zero_grad()
        
        # Batch processing
        batch_loss = 0
        size_of_batch = inputs.shape[0]
        predictions = []
        for i in range(0, size_of_batch):
            res_size = results[i].size()
            inp_size = inputs[i].size()
            decim_row = res_size[0] // inp_size[0]
            decim_col = res_size[1] // inp_size[1]
            prediction = model(inputs[i], decim_row, decim_col)
            predictions.append(prediction.unsqueeze(0))
            #loss: torch.Tensor = criterion(prediction, results[i])
            loss: torch.Tensor= sum([ criterion(hidden_f, results[i]) for hidden_f in model.hidden_states ])
            batch_loss += loss.item()
            loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
        batch_loss /= size_of_batch
        optimizer.step()
        
        # batch_loss /= size_of_batch
        
        if not(lr_scheduler is None):
            lr_scheduler.step()

        # Move prediction on datas device for Evaluator
        predictions = torch.concat(predictions, axis=0)
        predictions = predictions.to(datas_device, non_blocking=True)
        
        # Batch return on datas device
        inputs = inputs.to(datas_device, non_blocking=True)
        results = results.to(datas_device, non_blocking=True)
        
        output = {
            'prediction' : predictions,
            'result' : results,
            'loss' : batch_loss
        }

        
        return output

    return train_step


def update_loss_history(engine: ignite.engine.Engine, loss_history: list):
    loss_history.append(engine.state.output['loss'])

def save_loss_history(engine: ignite.engine.Engine, loss_history: list, loss_path: pathlib.Path):
    loss = numpy.array(loss_history)
    numpy.save(loss_path, loss)

def print_logs(engine: ignite.engine.Engine):
    strp = 'Epoch [{}/{}] : Loss {:.6f}'
    print(
        strp.format(
            engine.state.epoch,
            engine.state.max_epochs,
            engine.state.output['loss']
        )
    )

def save_model(
    engine: ignite.engine.Engine, 
    model: torch.nn.Module, 
    path: pathlib.Path = pathlib.Path('.')
) -> None:
    no_epoch = engine.state.epoch
    torch.save(model.state_dict(), path / 'model_epoch_{}.pt'.format(no_epoch))