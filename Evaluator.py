import torch.nn
import torch.utils.data

import torchvision.utils
import numpy

import ignite.engine

import pathlib

#from Unfolding2D import \
#    Datas
import Datas

def create_evaluate_function(
    model: torch.nn.Module,
    model_device: torch.device,
    datas_device: torch.device
):

    # model, optimizer, criterion, lr_scheduler = initialize(config)
    # Define any evaluation
    def eval_step(engine, batch):
        

        model.eval() # model.train(False)
        
        with torch.no_grad():

            inputs, results, _ = batch

            # Move batch on model_device
            inputs = inputs.to(model_device, non_blocking=True)
            results = results.to(model_device, non_blocking=True)

            predictions = []

            for i in range(0, inputs.shape[0]):
                res_size = results[i].size()
                inp_size = inputs[i].size()
                decim_row = res_size[0] // inp_size[0]
                decim_col = res_size[1] // inp_size[1]
                prediction = model(inputs[i], decim_row, decim_col)
                predictions.append(prediction.unsqueeze(0))

            # Move prediction on datas device for Evaluator
            predictions = torch.concat(predictions, axis=0)
            predictions = predictions.to(datas_device, non_blocking=True)

            # Batch return on datas device
            inputs = inputs.to(datas_device, non_blocking=True)
            results = results.to(datas_device, non_blocking=True)

            output = {
                'prediction' : predictions, 
                'result' : results
            }

            return output

    return eval_step


def update_history_metrics(
    engine: ignite.engine.Engine, 
    evaluator: ignite.engine.Engine,
    dataloader: torch.utils.data.DataLoader,
    history: dict[str, list],
) -> None:

    evaluator.run(dataloader, max_epochs=1)

    # no_epoch = engine.state.epoch
    
    # metrics = evaluator.state.metrics
    # mae = metrics['mae']
    # avg_mae = metrics['avg_mae']
    # mse = metrics['mse']
    # avg_mse = metrics['avg_mse']
    

    # # Print logs
    # str_print = mode + ' Results - Epoch {} - mae: {:.2f} Avg mae: {:.2f} mse: {:.2f} Avg mse: {:.2f}\n'
    # print(str_print.format(no_epoch, mae, avg_mae, mse, avg_mse))

    # Update history
    for key in evaluator.state.metrics.keys():
        history[key].append(evaluator.state.metrics[key])
        

def evaluate_dataloader(
    engine: ignite.engine.Engine,
    evaluator: ignite.engine.Engine,
    model: torch.nn.Module,
    model_device: torch.device,
    datas_device: torch.device,
    dataloader: Datas.ImageDataset,
    path_eval: pathlib.Path
) -> None:
    
    model.eval() # model.train(False)
        
    with torch.no_grad():
        
        no_epoch = engine.state.epoch
        imgs_output_path = path_eval / 'epoch_{}'.format(no_epoch)

        if not(imgs_output_path.exists()):
            imgs_output_path.mkdir()

        for batch in dataloader:

            inputs, results, filename = batch

            # Move batch on model_device
            inputs = inputs.to(model_device, non_blocking=True)
            results = results.to(model_device, non_blocking=True)

            model.eval() # model.train(False)

            for i in range(0, inputs.shape[0]):
                res_size = results[i].size()
                inp_size = inputs[i].size()
                decim_row = res_size[0] // inp_size[0]
                decim_col = res_size[1] // inp_size[1]
                prediction = model(inputs[i], decim_row, decim_col)
                prediction = prediction.cpu()
                torchvision.utils.save_image(prediction, imgs_output_path / (filename[i]+'.png'))
                numpy.save(imgs_output_path / (filename[i]+'.npy'), prediction.detach().numpy())

            # Batch return on datas device
            inputs = inputs.to(datas_device, non_blocking=True)
            results = results.to(datas_device, non_blocking=True)


def create_evaluate_function_with_variable_size_image(
    model: torch.nn.Module,
    model_device: torch.device,
    datas_device: torch.device
):

    # model, optimizer, criterion, lr_scheduler = initialize(config)
    # Define any evaluation
    def eval_step(engine, batch):
        

        model.eval() # model.train(False)
        
        with torch.no_grad():

            
            inputs, results = batch[0], batch[1]


            predictions = []

            for i in range(0, len(inputs)):

                res_size = results[i].size()
                inp_size = inputs[i].size()
                decim_row = res_size[0] // inp_size[0]
                decim_col = res_size[1] // inp_size[1]

                # to model device
                x = inputs[i].to(model_device, non_blocking=True)
                # y = results[i].to(model_device, non_blocking=True)

                
                prediction = model(x, decim_row, decim_col)
                predictions.append(prediction.unsqueeze(0).to(datas_device))

                # return on datas device
                inputs[i] = x.to(datas_device, non_blocking=True)
                # results[i] = y.to(datas_device, non_blocking=True)

            # Move prediction on datas device for Evaluator
              predictions = torch.concat(predictions, axis=0)
            # predictions = predictions.to(datas_device, non_blocking=True)

            # Batch return on datas device
            # inputs = inputs.to(datas_device, non_blocking=True)
            # results = results.to(datas_device, non_blocking=True)
              results = torch.stack(results)

            output = {
                'prediction' : predictions, 
                'result' : results
            }

            return predictions,results

    return eval_step



def evaluate_dataloader_with_variable_size_image(
    engine: ignite.engine.Engine,
    evaluator: ignite.engine.Engine,
    model: torch.nn.Module,
    model_device: torch.device,
    datas_device: torch.device,
    dataloader: Datas.ImageDataset,
    path_eval: pathlib.Path
) -> None:
    
    model.eval() # model.train(False)
        
    with torch.no_grad():
        
        no_epoch = engine.state.epoch
        imgs_output_path = path_eval / 'epoch_{}'.format(no_epoch)

        if not(imgs_output_path.exists()):
            imgs_output_path.mkdir()

        for batch in dataloader:

            inputs, results, filename = batch

            # Move batch on model_device
            # inputs = inputs.to(model_device, non_blocking=True)
            # results = results.to(model_device, non_blocking=True)

            model.eval() # model.train(False)

            for i in range(0, len(inputs)):

                res_size = results[i].size()
                inp_size = inputs[i].size()
                decim_row = res_size[0] // inp_size[0]
                decim_col = res_size[1] // inp_size[1]

                x = inputs[i].to(model_device, non_blocking=True)

                prediction = model(x, decim_row, decim_col)
                prediction = prediction.cpu()
                torchvision.utils.save_image(prediction, imgs_output_path / (filename[i]+'.png'))
                numpy.save(imgs_output_path / (filename[i]+'.npy'), prediction.detach().numpy())

                inputs[i] = x.to(datas_device, non_blocking=True)
            # Batch return on datas device
            # inputs = inputs.to(datas_device, non_blocking=True)
            # results = results.to(datas_device, non_blocking=True)
