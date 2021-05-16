import torch
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from models.graph_conv_model import GraphConvModel
from dataset import SmilesDataLoader, MoleculeDataset

from models.mol_graph import gcn_collate_fn


def print_result(trainer, task="regression"):
    print("train loss={0}".format(trainer.callback_metrics['train_loss']))
    if task == "regression":
        print("train r2={0}".format(trainer.callback_metrics['train_r2'].item()))
    else:
        print("train acc={0}".format(trainer.callback_metrics['train_acc'].item()))
        print("train rocauc={0}".format(trainer.callback_metrics['train_rocauc'].item()))

    print("validation loss={0}".format(trainer.callback_metrics['val_loss']))
    if task == "regression":
        print("validation r2={0}".format(trainer.callback_metrics['val_r2'].item()))
    else:
        print("validation acc={0}".format(trainer.callback_metrics['val_acc'].item()))
        print("validation rocauc={0}".format(trainer.callback_metrics['val_rocauc'].item()))


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu = 1
    else:
        device = torch.device('cpu')
        gpu = 0

    sdl = SmilesDataLoader("input/Lipophilicity.csv", "smiles", ["exp"])
    X_train, X_test, y_train, y_val = train_test_split(
            sdl.mols, 
            sdl.labels_list, 
            shuffle=True, 
            train_size=0.8, 
            random_state=226
            )
    md_train = MoleculeDataset(X_train, y_train)
    md_test = MoleculeDataset(X_test, y_val)

    data_loader_train = data.DataLoader(
            md_train, 
            batch_size=128, 
            shuffle=False,
            collate_fn=gcn_collate_fn
            )

    data_loader_val = data.DataLoader(
            md_test, 
            batch_size=128, 
            shuffle=False,
            collate_fn=gcn_collate_fn
            )

    model = GraphConvModel(
            device_ext=device,
            task="regression",
            conv_layer_sizes=[20, 20, 20],
            fingerprints_size=50,
            mlp_layer_sizes=[100, 1],
            lr=0.01
            )

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
            monitor='val_' + 'r2',
            dirpath='output',
            filename="model-{epoch:02d}-{val_" + 'r2' +":.5f}",
            save_top_k=3,
            mode='max'
    )
    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
            max_epochs=30,
            gpus=gpu,
            callbacks=callbacks
    )
    trainer.fit(model, data_loader_train, data_loader_val)
    print_result(trainer)


if __name__ == "__main__":
    main()
