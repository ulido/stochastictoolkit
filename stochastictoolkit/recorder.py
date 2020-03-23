import numpy as np
import pandas as pd
import tables
import pathlib
from collections import namedtuple
import pickle

import logging

logger = logging.getLogger(__name__)

class Recorder:
    def __init__(self, filename):
        self._parameters = {}
        self._frozen = False
        self._recording_types = {}
        self._recording_types_under_construction = {}
        self._arrays_to_save = {}

        if pathlib.Path(filename).exists():
            raise ValueError(f'File {filename} exists!')
        self._filename = filename
        
        self.__logger = logger.getChild('Recorder')
        
    def register_parameter(self, name, value):
        if self._frozen:
            raise RuntimeError("Trying to register parameter after recording has started!")
        if name in self._parameters:
            raise KeyError(f"Parameter {name} was already registered!")
        self.__logger.info(f"Registering parameter {name}")
        self._parameters[name] = value

    def register_parameters(self, parameters):
        for k, v in parameters.items():
            self.register_parameter(k, v)

    def record_array(self, name, array, index_value=None):
        # Saving arrays is done instantaneously rather than waiting until save is called.
        # Otherwise we might run into memory issues.
        with tables.open_file(self._filename, mode='a') as h5file:
            if name not in h5file.root:
                data = h5file.create_earray(f'/{name}', 'data', obj=array[np.newaxis], createparents=True)
                if index_value is not None:
                    index = h5file.create_earray(f'/{name}', 'index', obj=np.array(index_value)[np.newaxis])
            else:
                group = h5file.get_node(f'/{name}')
                if 'index' in group:
                    if index_value is None:
                        raise ValueError('Index present but no index value specified!')
                    group['index'].append(np.array(index_value)[np.newaxis])
                    group['data'].append(array[np.newaxis])
            
    def new_recording_type(self, name, fields):
        self.__logger.info(f"Registering recording type {name}")
        if self._frozen:
            raise RuntimeError("Trying to create type after recording has started!")            
        self._recording_types_under_construction[name] = fields

    def _build_recording_types(self):
        self.__logger.info(f"Building recording types")
        self._frozen = True
        for name, fields in self._recording_types_under_construction.items():
            self._recording_types[name] = (namedtuple(name, list(fields)), [])

    def record(self, type_name, **items):
        if not self._frozen:
            self._build_recording_types()
        rec_type, rows = self._recording_types[type_name]
        self.__logger.info(f"Recording event of type {type_name}")
        rows.append(rec_type(**items))

    def save(self):
        if not self._frozen:
            self._build_recording_types()
        df = pd.DataFrame([{'parameter': k,
                            'value': str(repr(v)),
                            'pickled_value': str(pickle.dumps(v))}
                           for k, v in self._parameters.items()])
        df.to_hdf(self._filename, mode='a', key='parameters')
        for type_name, (_, rows) in self._recording_types.items():
            self.__logger.info(f"Saving {len(rows)} recorded events for type {type_name}")
            df = pd.DataFrame(rows)
            df.to_hdf(self._filename, mode='a', key=type_name)

