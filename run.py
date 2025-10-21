
import Evaluator
import Model
import Datas

import torch.optim
import torch.nn
import torch.cuda
import torch.utils.data
import torch.autograd

import ignite.engine
import ignite.metrics
import ignite.contrib.handlers

import pathlib

import pandas
import numpy
import yaml
import sys

torch.autograd.set_detect_anomaly(True)


def read_config(path: pathlib.Path) -> dict:
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_config(config: dict, path: pathlib.Path) -> None:
    with open(path, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

# import Trainer
# def train_from_config(
#         config: dict, 
#         train_folder: pathlib.Path('.')
# ) -> None :
#     # # Read config
#     # train_folder = pathlib.Path(sys.argv[1])
#     # config = read_config(train_folder / 'config.yml')
    
#     # Make outputs paths
#     # if not(parent_output.exists()):
#     #     parent_output.mkdir()

#     # output_path = parent_output
#     output_path = pathlib.Path(config['output'].get('folder', train_folder))
#     if not(output_path.exists()):
#         output_path.mkdir()

#     models_save_path = output_path / config['output']['models_save']['path']
#     if not(models_save_path.exists()):
#         models_save_path.mkdir()
#     models_save_every = config['output']['models_save']['every']
    
#     imgs_save_path = output_path / config['output']['imgs_save']['path']
#     if not(imgs_save_path.exists()):
#         imgs_save_path.mkdir()

#     path_imgs_train = imgs_save_path / 'train_datas'
#     if not(path_imgs_train.exists()):
#         path_imgs_train.mkdir()

#     path_imgs_eval = imgs_save_path / 'eval_datas'
#     if not(path_imgs_eval.exists()):
#         path_imgs_eval.mkdir()
    
#     imgs_save_every = config['output']['imgs_save']['every']

#     df_training_path = output_path / config['output']['metrics']['train']
#     df_validation_path = output_path / config['output']['metrics']['validation']
#     loss_path = output_path / config['output']['loss']

#     # Dataset params
#     dataset_path = pathlib.Path(config['dataset']['path'])
#     datas_device = config['dataset']['device']
#     batch_size = config['dataset']['params']['batch_size']
#     train_size = config['dataset']['params']['train_size']
#     #datas_shuffle = config['dataset']['params']['shuffle']

#     # Model params
#     model_device = config['model']['device']
#     # nb_iteration = config['model']['params']['nb_iteration']
#     # nb_channel = config['model']['params']['nb_channel']
#     # kernel_size = config['model']['params']['kernel_size']

#     # Training params
#     nb_epochs = config['train']['nb_epochs']
#     learning_rate = config['train']['learning_rate']
#     criterion = eval(config['train']['loss'])

#     clip_value_using = 'gradient_clip_value' in config['train'].keys()
#     if clip_value_using:
#         clip_value = config['train']['gradient_clip_value']

#     #################################################################
    
#     # Make Dataset and Dataloaders

#     dataset_full = Datas.ImageDataset(dataset_path, datas_device)
#     dataset_train, dataset_validation = Datas.split_dataset(dataset_full, train_size=train_size)


#     #dataset_train = dataset_train.to(datas_device, non_blocking=True)
#     #dataset_validation = dataset_validation.to(datas_device, non_blocking=True)

#     dataloader_train = torch.utils.data.DataLoader(
#         dataset_train, 
#         batch_size=batch_size,
#         collate_fn = Datas.get_batch_with_variable_size_image,
#         shuffle=True
#     )


#     dataloader_validation= torch.utils.data.DataLoader(
#         dataset_validation, 
#         batch_size=batch_size,
#         collate_fn = Datas.get_batch_with_variable_size_image,
#         shuffle=True,
#     )
    
#     # Make Trainer
    
#     output_transform = \
#         lambda output: (output['prediction'], output['result'])

#     # model = Model.Unfolding(nb_channel, kernel_size, nb_iteration)
#     model = Model.Unfolding.from_config(config)

#     if clip_value_using:
#         for p in model.parameters():
#             p.register_hook(
#                 lambda grad: torch.clamp(grad, -clip_value, clip_value)
#             )

#     optimizer = torch.optim.Adam(
#         params=model.parameters(),
#         lr = learning_rate
#     )

#     # criterion = ignite.metrics.MeanAbsoluteError(output_transform)
#     #criterion = ignite.metrics.MeanAbsoluteError(output_transform)
#     criterion = torch.nn.MSELoss()

#     lr_scheduler = None

#     # model.to(device=...)

#     model = model.to(model_device)
#     train_step = Trainer.create_train_step_with_variable_size_image(
#         model, model_device, datas_device, optimizer, criterion, lr_scheduler
#     )

#     trainer = ignite.engine.Engine(train_step)


#     # Make evaluator
#     evaluate_function = Evaluator.create_evaluate_function_with_variable_size_image(model, model_device, datas_device)
#     evaluator = ignite.engine.Engine(evaluate_function)

#     #### MAE METRICS

#     # mae = ignite.metrics.MeanAbsoluteError(output_transform)
#     # avg_mae = ignite.metrics.RunningAverage(src=mae, epoch_bound=False)

#     # mae.attach(engine=evaluator, name='mae')
#     # avg_mae.attach(engine=evaluator, name='avg_mae')

#     #### MSE METRICS

#     # mse = ignite.metrics.MeanSquaredError(output_transform)
#     # avg_mse = ignite.metrics.RunningAverage(src=mse, epoch_bound=False)

#     # mse.attach(engine=evaluator, name='mse')
#     # avg_mse.attach(engine=evaluator, name='avg_mse')

#     criterion_metric = ignite.metrics.Loss(
#         criterion,
#         output_transform
#     )

#     avg_criterion_metric = ignite.metrics.RunningAverage(
#         src=criterion_metric,
#         epoch_bound=True
#     )


#     criterion_metric.attach(
#         engine=evaluator, 
#         name='loss'
#     )
#     avg_criterion_metric.attach(
#         engine=evaluator, 
#         name='avg_loss'
#     )
    
#     #### History
    
#     validation_history = {
#         'loss' : [],
#         'avg_loss' : []
#     }

#     training_history = {
#        'loss' : [],
#         'avg_loss' : []
#     }

        
#     loss_history = []
    
#     #### Event handler
    
#     trainer.add_event_handler(
#         ignite.engine.Events.EPOCH_COMPLETED,
#         # Callback
#         Trainer.update_loss_history,
#         # Parameters of callback
#         loss_history
#     )

#     trainer.add_event_handler(
#         ignite.engine.Events.EPOCH_COMPLETED,
#         # Callback
#         Trainer.save_loss_history,
#         # Parameters of callback
#         loss_history, 
#         loss_path
#     )
    
#     trainer.add_event_handler(
#         # ignite.engine.Events.COMPLETED,
#         ignite.engine.Events.EPOCH_COMPLETED(every=models_save_every)
#         | ignite.engine.Events.COMPLETED,
#         # Callback
#         Trainer.save_model,
#         # Parameters of callback
#         model,
#         models_save_path
#     )

#     #trainer.add_event_handler(
#     #    ignite.engine.Events.EPOCH_COMPLETED,
#     #    # Callback
#     #    Trainer.print_logs
#     #)

#     ## Evaluation on datas using for training
#     trainer.add_event_handler(
#         ignite.engine.Events.EPOCH_COMPLETED,
#         # Callback
#         Evaluator.update_history_metrics,
#         # Parameters of callback
#         evaluator, 
#         dataloader_train, 
#         training_history
#     )

#     ## Evaluation on datas using for validation
#     trainer.add_event_handler(
#         ignite.engine.Events.EPOCH_COMPLETED,
#         # Callback
#         Evaluator.update_history_metrics,
#         # Parameters of callback
#         evaluator, 
#         dataloader_validation, 
#         validation_history
#     )
    
#     trainer.add_event_handler(
#         ignite.engine.Events.EPOCH_COMPLETED(every=imgs_save_every)
#         | ignite.engine.Events.COMPLETED,
#         # Callback
#         Evaluator.evaluate_dataloader_with_variable_size_image,
#         # Parameters of callback
#         evaluator,
#         model,
#         model_device,
#         datas_device,
#         dataloader_train,
#         path_imgs_train
#     )

#     trainer.add_event_handler(
#         ignite.engine.Events.EPOCH_COMPLETED(every=imgs_save_every)
#         | ignite.engine.Events.COMPLETED,
#         # Callback
#         Evaluator.evaluate_dataloader_with_variable_size_image,
#         # Parameters of callback
#         evaluator,
#         model,
#         model_device,
#         datas_device,
#         dataloader_validation,
#         path_imgs_eval
#     )
    
    
#     # Run model
    
#     _ = trainer.run(dataloader_train, max_epochs=nb_epochs)
    
#     # # Save metrics
    
#     df_training = pandas.DataFrame(training_history)
#     df_validation = pandas.DataFrame(validation_history)

#     df_training.to_pickle(df_training_path)
#     df_validation.to_pickle(df_validation_path)

    
#     # Save loss
#     loss = numpy.array(loss_history)
#     numpy.save(loss_path, loss)

import CustomTrainer
def train_from_config(
    config: dict, 
    train_folder: pathlib.Path('.')
) -> None :
    
    # Make outputs paths
    output_path = pathlib.Path(config['output'].get('folder', train_folder))
    if not(output_path.exists()):
        output_path.mkdir()

    models_save_path = output_path / config['output']['models_save']['path']
    if not(models_save_path.exists()):
        models_save_path.mkdir()
    models_save_every = config['output']['models_save']['every']
    
    imgs_save_path = output_path / config['output']['imgs_save']['path']
    if not(imgs_save_path.exists()):
        imgs_save_path.mkdir()

    path_imgs_train = imgs_save_path / 'train_datas'
    if not(path_imgs_train.exists()):
        path_imgs_train.mkdir()

    path_imgs_eval = imgs_save_path / 'eval_datas'
    if not(path_imgs_eval.exists()):
        path_imgs_eval.mkdir()
    
    imgs_save_every = config['output']['imgs_save']['every']

    df_training_path = output_path / config['output']['metrics']['train']
    df_validation_path = output_path / config['output']['metrics']['validation']
    loss_path = output_path / config['output']['loss']


    # Dataset params
    dataset_path = pathlib.Path(config['dataset']['path'])
    datas_device = config['dataset']['device']
    batch_size = config['dataset']['params']['batch_size']
    train_size = config['dataset']['params']['train_size']
    #datas_shuffle = config['dataset']['params']['shuffle']

    # Model params
    model_device = config['model']['device']
    # nb_iteration = config['model']['params']['nb_iteration']
    # nb_channel = config['model']['params']['nb_channel']
    # kernel_size = config['model']['params']['kernel_size']

    # Training params
    nb_epochs = config['train']['nb_epochs']
    learning_rate = config['train']['learning_rate']
    criterion = eval(config['train']['loss'])

    clip_value_using = 'gradient_clip_value' in config['train'].keys()
    if clip_value_using:
        clip_value = config['train']['gradient_clip_value']

    #################################################################
    
    # Make Dataset and Dataloaders

    dataset_full = Datas.ImageDataset(dataset_path, datas_device)
    dataset_train, dataset_validation = Datas.split_dataset(dataset_full, train_size=train_size)


    #dataset_train = dataset_train.to(datas_device, non_blocking=True)
    #dataset_validation = dataset_validation.to(datas_device, non_blocking=True)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=batch_size,
        collate_fn = Datas.get_batch_with_variable_size_image,
        shuffle=True
    )


    dataloader_validation= torch.utils.data.DataLoader(
        dataset_validation, 
        batch_size=batch_size,
        collate_fn = Datas.get_batch_with_variable_size_image,
        shuffle=True,
    )
    
    # Make Trainer
    
    output_transform = lambda output: (output[0], output[1])
    
    # model = Model.Unfolding(nb_channel, kernel_size, nb_iteration)
    model = Model.Unfolding.from_config(config)

    if clip_value_using:
        for p in model.parameters():
            p.register_hook(
                lambda grad: torch.clamp(grad, -clip_value, clip_value)
            )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr = learning_rate
    )

    # criterion = ignite.metrics.MeanAbsoluteError(output_transform)
    #criterion = ignite.metrics.MeanAbsoluteError(output_transform)
    criterion = torch.nn.MSELoss()

    # model.to(device=...)

    model = model.to(model_device)
    train_step = CustomTrainer.create_train_step(
        model, model_device, datas_device, optimizer, criterion
    )

    trainer = CustomTrainer.CustomEngine(train_step)
    trainer.add_event_handler(
        ignite.engine.Events.ITERATION_COMPLETED,
        CustomTrainer.update_epoch_loss
    )
    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED,
        CustomTrainer.compute_epoch_loss
    )
    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED,
        CustomTrainer.save_epoch_loss,
        loss_path
    )
    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED,
        CustomTrainer.print_logs
    )


    # Make evaluator
    evaluate_function = Evaluator.create_evaluate_function_with_variable_size_image(model, model_device, datas_device)
    evaluator = ignite.engine.Engine(evaluate_function)

    #### MAE METRICS

    mae = ignite.metrics.MeanAbsoluteError(output_transform)
    avg_mae = ignite.metrics.RunningAverage(src=mae, epoch_bound=True)

    mae.attach(engine=evaluator, name='mae')
    avg_mae.attach(engine=evaluator, name='avg_mae')

    #### MSE METRICS

    mse = ignite.metrics.MeanSquaredError(output_transform)
    avg_mse = ignite.metrics.RunningAverage(src=mse, epoch_bound=True)

    mse.attach(engine=evaluator, name='mse')
    avg_mse.attach(engine=evaluator, name='avg_mse')
    
    #### History
    
    validation_history = {
        'mae' : [],
        'avg_mae' : [],
        'mse' : [],
        'avg_mse' : []
    }

    training_history = {
        'mae' : [],
        'avg_mae' : [],
        'mse' : [],
        'avg_mse' : [],
    }

        
    loss_history = []
    
    #### Event handler
    
    # trainer.add_event_handler(
    #     ignite.engine.Events.EPOCH_COMPLETED,
    #     # Callback
    #     Trainer.update_loss_history,
    #     # Parameters of callback
    #     loss_history
    # )

    # trainer.add_event_handler(
    #     ignite.engine.Events.EPOCH_COMPLETED,
    #     # Callback
    #     Trainer.save_loss_history,
    #     # Parameters of callback
    #     loss_history, 
    #     loss_path
    # )
    
    trainer.add_event_handler(
        # ignite.engine.Events.COMPLETED,
        ignite.engine.Events.EPOCH_COMPLETED(every=models_save_every) 
        | ignite.engine.Events.COMPLETED,
        # Callback
        CustomTrainer.save_model,
        # Parameters of callback
        model,
        models_save_path
    )

    # trainer.add_event_handler(
    #     # ignite.engine.Events.COMPLETED,
    #     ignite.engine.Events.COMPLETED,
    #     # Callback
    #     Trainer.save_model,
    #     # Parameters of callback
    #     model,
    #     models_save_path
    # )

    #trainer.add_event_handler(
    #    ignite.engine.Events.EPOCH_COMPLETED,
    #    # Callback
    #    Trainer.print_logs
    #)

    ## Evaluation on datas using for training
    #trainer.add_event_handler(
        # ignite.engine.Events.EPOCH_COMPLETED,
        # Callback
        # Evaluator.update_history_metrics,
        # Parameters of callback
        # evaluator, 
        # dataloader_train, 
        # training_history
    #)

    ## Evaluation on datas using for validation
    #trainer.add_event_handler(
     #   ignite.engine.Events.EPOCH_COMPLETED,
        # Callback
     #  Evaluator.update_history_metrics,
        # Parameters of callback
     #   evaluator, 
     #  dataloader_validation, 
     #   validation_history
    #)
    
    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED(every=imgs_save_every) 
        | ignite.engine.Events.COMPLETED,
        # Callback
        Evaluator.evaluate_dataloader_with_variable_size_image,
        # Parameters of callback
        evaluator,
        model,
        model_device,
        datas_device,
        dataloader_train,
        path_imgs_train
    )

    trainer.add_event_handler(
        ignite.engine.Events.EPOCH_COMPLETED(every=imgs_save_every)
        | ignite.engine.Events.COMPLETED,
        # Callback
        Evaluator.evaluate_dataloader_with_variable_size_image,
        # Parameters of callback
        evaluator,
        model,
        model_device,
        datas_device,
        dataloader_validation,
        path_imgs_eval
    )
    
    
    # Run model
    
    _ = trainer.run(dataloader_train, max_epochs=nb_epochs)
    
    
    # # Save metrics
    
    df_training = pandas.DataFrame(training_history)
    df_validation = pandas.DataFrame(validation_history)

    df_training.to_pickle(df_training_path)
    df_validation.to_pickle(df_validation_path)

if __name__ == '__main__' :

    # Read config
    train_folder = pathlib.Path(sys.argv[1])
    config = read_config(train_folder / 'config.yml')
    
    train_from_config(
        config = config,
        train_folder = train_folder
    )
