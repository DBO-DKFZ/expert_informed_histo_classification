import re
from pathlib import Path
from typing import Mapping, Optional

import h5py
import lightning as L
import numpy as np
import torch
import torchvision
from torchvision.transforms import v2

from datasets import SimpleImageDatasetV2
from lit_datamodule import LitDataModule
from lit_models import LitModule
from utils.custom_types import FilePath


def get_neighbours(tile: tuple[int,int], tiles: dict[tuple[int,int]: int]) -> list[int]:
    """
    Finds the indices of all neighbouring tiles for specific tile. Neighbouring also includes over corner adjacency.

    :param tile: the tile (x,y) to find the neighbours for
    :param tiles: a list containing all tiles in the format (x,y): idx
    :return: a list of indices of the neighbouring tiles.
    """
    x, y = tile
    moves = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    return [tiles[(x + dx, y + dy)] for dx, dy in moves if (x + dx, y + dy) in tiles]


def create_adj(tiles: list[FilePath], parse: callable = lambda x: x.stem.split('_')) -> tuple[list[int], list[int]]:
    """
    Creates the adjacency indices for all tiles.

    :param tiles: a list containing all tiles (Paths, names, etc.)
    :param parse: a function to parse the tile names to the format (x,y). Default assumes a Path like: <directories>/<x>_<y>.<filetype>
    :return: an adjacency matrix in sparse format (as a tuple of two lists)
    """
    # convert tile names to (x,y): idx
    tiles = {tuple(map(int, parse(tile))): idx for idx, tile in enumerate(tiles)}

    t0, t1 = [], []
    # check for each tile...
    for tile in tiles:
        # ...indices of neighbouring tiles...
        neighbours = get_neighbours(tile, tiles)
        # ...then append edges
        t0 += [tiles[tile] for _ in range(len(neighbours))]
        t1 += neighbours

    return t0, t1


def load_model(model_path: FilePath, model_name: str = 'resnet18') -> torch.nn.Module:
    """
    Loads a torchvision model with weights from a model checkpoint

    :param model_path: path to model checkpoint
    :param model_name: name of the model
    :return: a model
    """
    model = torchvision.models.__dict__[model_name](weights=None)

    state = torch.load(model_path, map_location='cuda:0')

    state_dict = state['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

    model_dict = model.state_dict()
    weights = {k: v for k, v in state_dict.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    model.fc = torch.nn.Sequential()

    model = model.cuda()

    return model


def create_graphs(in_path: FilePath, out_path: FilePath, model_path: FilePath, model_name: str = 'resnet18', filetype: str = 'jpeg',
                  labels: Optional[Mapping[str, int]] = None, names: bool = False,
                  parse_slide: callable = lambda x: x, parse_tile: callable = lambda x: x.stem.split('_')) -> None:
    """
    Creates graphs for each slide from their tiles and saves as an HDF5.
    Expects the following directory structure: <in_path>/<slide>/.../<tile.filetype>

    :param in_path: path to slides.
    :param out_path: where to store the hdf5 (including filename)
    :param model_path: where the model backbone is stored
    :param model_name: the model's name (i.e. torchvision.models)
    :param filetype: the tiles filetype
    :param labels: a mapping of slide names to labels
    :param names: whether to save the slide names
    :param parse_slide: a function to parse the slide name
    :param parse_tile: a function to parse the tile names to the format (x,y). Default assumes a Path like: <directories>/<x>_<y>.<filetype>
    :return:
    """
    if not (labels or names):
        print('WARNING: Neither labels nor names are set.')
    slides = [d for d in Path(in_path).iterdir() if d.is_dir()]
    transform_test = v2.Compose([v2.ToImage(), v2.Resize([224,224], antialias=True), v2.ToDtype(torch.float32, scale=True)])

    # create model & trainer
    model = LitModule(load_model(model_path=model_path, model_name=model_name), None, None)
    trainer = L.Trainer(devices=1)

    print(f'Creating {out_path}...')
    with h5py.File(out_path, 'w') as hdf5_file:
        # initialize all datasets
        length = len(slides)
        if labels:
            hdf5_file.create_dataset('labels', data=[labels[parse_slide(slide.stem)] for slide in slides])
        if names:
            hdf5_file.create_dataset('names', data=[parse_slide(slide.stem) for slide in slides])
        feature_dataset = hdf5_file.create_dataset('features', shape=(length, 512), dtype=h5py.vlen_dtype(np.float32))
        adjacency_dataset = hdf5_file.create_dataset('adjacency', shape=(length, 2), dtype=h5py.vlen_dtype(np.int64))
        pos_dataset = hdf5_file.create_dataset('position', shape=(length, 2), dtype=h5py.vlen_dtype(np.int64))

        # for all slides...
        for idx, slide in enumerate(slides):
            # get paths to tiles
            tiles = sorted(slide.rglob(f'*{filetype}'))
            # create and save features
            datamodule = LitDataModule(data_set=SimpleImageDatasetV2, data_dir=tiles, transform_train=None, transform_test=transform_test, split='')
            features = trainer.predict(model=model, datamodule=datamodule)
            feature_dataset[idx] = torch.vstack([f[0] for f in features]).numpy().T
            # create and save adjacency matrix
            adj = create_adj(tiles, parse=parse_tile)
            adjacency_dataset[idx] = np.array(adj)
            # create and save positions
            pos = [parse_tile(tile) for tile in tiles]
            pos_dataset[idx] = np.array(pos).T


create_graphs(in_path='/path/to/folder/with/tiled/slides', 
              out_path='/path/to/outputfile.hdf5',
              model_path='/path/to/backbone',
              model_name='resnet18',
              filetype='jpeg',
              labels=None, # Add your slide to labels mapping here (if applicable)
              parse_slide=lambda x: x,
              parse_tile=lambda x: (x.stem.split('_')[0], x.stem.split('_')[1]),
              names=True)

