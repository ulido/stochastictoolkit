'''recorder.py

Infrastructure for recording parameters and simulation data

Classes
-------
Recorder: Class for recording and saving parameters, events and data
'''

import numpy as np
import pandas as pd
import tables
import pathlib
from collections import namedtuple
import pickle

__all__ = ['Recorder']

class Recorder:
    '''Recorder class for recording and storing parameters, events and data'''

    def __init__(self, filename, batch_size=10000):
        '''Initialize the Recorder object

        A ValueError is raised if the given file already exists.

        Parameters
        ----------
        filename: str
            Filename of the hdf5 database the data should be stored in.
        '''
        self._parameters = {}
        self._frozen = False
        self._recording_types = {}
        self._recording_types_under_construction = {}
        self._arrays_to_save = {}

        if pathlib.Path(filename).exists():
            raise ValueError(f'File {filename} exists!')
        self._filename = filename

        self.batch_size = batch_size
        self.store = pd.HDFStore(filename)
        
    def register_parameter(self, name, value):
        '''Register a single parameter name and value pair

        Raises a KeyError if the parameter name was already claimed.

        Parameter
        ---------
        name: str
            Parameter name
        value: object
            Parameter value
        '''
        if self._frozen:
            raise RuntimeError("Trying to register parameter after recording has started!")
        if name in self._parameters:
            raise KeyError(f"Parameter {name} was already registered!")
        self._parameters[name] = value

    def register_parameters(self, parameters):
        '''Batch register parameters

        Parameter
        ---------
        parameters: dict
            Dictionary of parameter key value pairs.
        '''
        for k, v in parameters.items():
            self.register_parameter(k, v)

    def record_array(self, name, array, index_value=None):
        '''Store an array.
        
        If the `name` does not exist in the database, a new HDF5
        EArray is created under the sub-key `/data`. If `index_value`
        is not `None`, a second HDF5 EArray is created under sub-key
        `/index`. If the `name` does exist the `array` shape and
        dtype, as well as the `index_val` needs to match.

        Saving arrays is done instanteously instead of waiting until
        save is called to avoid memory issues.

        Parameters
        ----------
        name: str
            Name of the array
        array: ndarray
            The array to store
        index_value: any numeric type
            The index of the array slice to store

        '''
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
        '''Create a new recording data type.

        Parameters
        ----------
        name: str
            Name of the new data type
        fields: sequence
            List of fields in this data type
        '''
        if self._frozen:
            raise RuntimeError("Trying to create type after recording has started!")            
        self._recording_types_under_construction[name] = fields

    def _build_recording_types(self):
        '''Finalize all data types'''
        self._frozen = True
        for name, fields in self._recording_types_under_construction.items():
            self._recording_types[name] = (namedtuple(name, list(fields)), [])

    def record(self, type_name, **items):
        '''Store a record

        All field names need to be present in items.

        Parameters
        ----------
        type_name: str
            Name of the data type to store
        **items: dict
            Contains the key-value pairs of the data to store
        '''
        if not self._frozen:
            self._build_recording_types()
        rec_type, rows = self._recording_types[type_name]
        rows.append(rec_type(**items))
        if len(rows) >= self.batch_size:
            self.store.append(type_name, pd.DataFrame(rows), format='table')
            rows.clear()

    def save(self):
        '''Save all records in the database file.'''
        if not self._frozen:
            self._build_recording_types()
        df = pd.DataFrame([{'parameter': k,
                            'value': str(repr(v)),
                            'pickled_value': str(pickle.dumps(v))}
                           for k, v in self._parameters.items()])
        self.store.put('parameters', df, format='table')

        for type_name, (_, rows) in self._recording_types.items():
            if len(rows) > 0:
                df = pd.DataFrame(rows)
                self.store.append(type_name, df, format='table')

        self.store.close()
