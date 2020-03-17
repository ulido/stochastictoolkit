import numpy as np

from stochastictoolkit.particle_type import Recorder

import tables
import pandas as pd

def get_filename_from_tmppath(path, fname):
    h5path = path / fname
    return h5path.resolve().as_posix()


def test_array(tmp_path):
    fname = get_filename_from_tmppath(tmp_path, 'arraytest.h5')
    recorder = Recorder(fname)
    times = np.arange(10)
    array = np.random.uniform(size=(times.shape[0], 10, 10))
    for i, time in enumerate(times):
        recorder.record_array('testarray', array[i], time)
    recorder.save()

    with tables.open_file(fname, 'r') as h5file:
        arrayobj = h5file.get_node('/testarray', 'data')
        indexobj = h5file.get_node('/testarray', 'index')
        read_array = arrayobj[:]
        read_times = indexobj[:]

    assert(abs(array - read_array).sum() < 1e-8)
    assert(abs(times - read_times).sum() < 1e-8)

def test_parameters(tmp_path):
    fname = get_filename_from_tmppath(tmp_path, 'arraytest.h5')
    recorder = Recorder(fname)

    test_p = ('parameter1', 1)
    recorder.register_parameter(*test_p)

    test_parameters = {
        'parameter2': 2,
        'parameter3': np.array([5, 6]),
    }
    recorder.register_parameters(test_parameters)

    test_parameters.update(dict([test_p]))
    parameter_names = list(test_parameters.keys())
    
    recorder.save()

    read_parameters = {row['parameter']: eval(row['value'], None, {'array': np.array})
                       for _, row in pd.read_hdf(fname, '/parameters').iterrows()}
    for k, v in read_parameters.items():
        assert(k in parameter_names)
        if type(v) == np.ndarray:
            assert((v == test_parameters[k]).all())
        else:
            assert(v == test_parameters[k])
        parameter_names.remove(k)
    assert(len(parameter_names) == 0)

def test_table(tmp_path):
    fname = get_filename_from_tmppath(tmp_path, 'arraytest.h5')
    recorder = Recorder(fname)

    recorder.new_recording_type('TestType', ['a', 'b', 'c'])

    recorder.record('TestType', a=1, b='B', c=0.5)

    recorder.save()

    df = pd.read_hdf(fname, '/TestType')

    assert(df.shape == (1, 3))

    row = df.iloc[0]
    assert(row.a == 1)
    assert(row.b == 'B')
    assert(row.c == 0.5)
