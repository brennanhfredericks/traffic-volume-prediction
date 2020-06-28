import h5py
import os
import random
import numpy as np
import more_itertools
import tensorflow as tf

class hdf5_manager:

    def __init__(self, filename, load=False):

        if not load:
            self.__root = h5py.File(f"{filename}.hdf5", 'x')
            self.__filename = f"{filename}.hdf5"
            self.__init()

        else:
            assert os.path.isfile(filename)
            self.__filename = filename
            self.__root = h5py.File(filename, 'r+')

    def __init(self):
        self.__root.create_group("links")
        self.__root.create_group("datetime")
        self.__root.create_group("masks")

        self.__maxshapes = {
            "datetime/datetime": (None, 1),
            "links/this": (None, 42),
            "masks/missing_this": (None, 1)
        }

    def __init_dataset(self, path, data):
        __path = path.split('/')
        self.__root[__path[0]].create_dataset(
            __path[1], data=data, maxshape=self.__maxshapes[path])

    def close(self):
        self.__root.close()

    def __flush(self):
        self.__root.flush()

    def write(self, path, data):
        """
        path -> the path of the dataset including parent -> links/this
        data -> nd-array 
        """
        try:
            self.__root[path].resize(
                self.__root[path].shape[0] + data.shape[0], axis=0)
            self.__root[path][-data.shape[0]:] = data
        except KeyError:
            self.__init_dataset(path, data)
        finally:
            self.__flush()

    def read(self, path):
        """
        path -> the path of the dataset including parent -> links/this
        """
        return self.__root[path]

    def __enter__(self):
        try:
            self.__root.mode
            return self.__root
        except ValueError:
            self.__root = h5py.File(self.__filename, 'r+')
            return self.__root

    def __exit__(self, *args, **kwargs):
        self.close()


class train_dev_test:

    def __init__(self, filepath, split=(80, 10, 10)):
        """
        filepath -> 
        """

        assert len(split) == 3 and sum(split) == 100
        self.__split = split

        self.__raw_data = hdf5_manager(filepath, load=True)

        self.__data_shape = self.__raw_data.read("links/this").shape

        self.__raw_data.close()

    def init(self, history_size=None, future_size=None, load_params=False):
        """
        history_size -> the number of previous timestemps that would serve as input to the model ->
                        X = (timestemp,features)
        future_size -> the number of timestemps (into the future) we trying to predict

        load_params ->  when False -> uses history and future size to select valide indices from the raw data (single_vds.hdf5), 
                        indices are then random selected into to train, dev and test sets. based on the train indices the mean and variance
                        parameters are calculated. the train, dev, test, mean and variance variables are store in the (single_ vds.hdf5) for reuse purposes.

                        to re initial the train, dev, test, mean and variance variables set load_params=False

                        when True -> the stored train, dev, test, mean and variance variables are loaded for reuse purposes
        """

        if not load_params:
            if history_size is None or future_size is None:
                raise ValueError("history or future size is None")

            self.history_size = history_size
            self.future_size = future_size

            print(
                "determining all valid indices - based on history and future size and missing data mask")
            self.__init_indices()
            print("randomly splitting indices into train, dev and test sets")
            self.__select_random_indices()  # select indices continuously?
            print("calculating mean and variance paramaters")
            self.__init_params()
            print("storing paramaters")
            self.__store_params()

        else:
            try:
                with self.__raw_data as f:
                    f['params'].values()
            except:
                raise ValueError(
                    "no parameters have been stored -> set load_params=False to initial")

            self.__load_params()

    @property
    def params(self):
        print(f"history_size:  {self.history_size}")
        print(f"future_size: {self.future_size}")
        print(f"mean: {self.mean}")
        print(f"variance: {self.variance}")
        print(f"train indices: {self.train_indices.shape}")
        print(f"dev indices: {self.dev_indices.shape}")
        print(f"test indices: {self.test_indices.shape}")

    def train_ds(self,batch_size=128,buffer_size=10000):

        __train_ds = tf.data.Dataset.from_generator(
            lambda: self.__sequence_generator(self.train_indices),
            (tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([self.history_size,self.__data_shape[-1]]),tf.TensorShape([self.future_size]))
        )

        return __train_ds.shuffle(buffer_size).batch(batch_size).cache()

    def dev_ds(self,batch_size=128,buffer_size=1000):
        __dev_ds = tf.data.Dataset.from_generator(
            lambda: self.__sequence_generator(self.dev_indices),
            (tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([self.history_size,self.__data_shape[-1]]),tf.TensorShape([self.future_size]))
        )

        return __dev_ds.shuffle(buffer_size).batch(batch_size).cache()

    def test_ds(self,batch_size=128):
        __test_ds = tf.data.Dataset.from_generator(
            lambda: self.__sequence_generator(self.test_indices),
            (tf.float32, tf.float32),
            output_shapes=(tf.TensorShape([self.history_size,self.__data_shape[-1]]),tf.TensorShape([self.future_size]))
        )

        return __test_ds.batch(batch_size)

    def __init_indices(self):

        self.__valid_indices = []
        __mask = np.zeros((self.history_size+self.future_size, 1))

        with self.__raw_data as data:
            for i in range(self.history_size, self.__data_shape[0]-self.future_size):
                data["masks/missing_this"].read_direct(__mask, source_sel=np.s_[
                                                       i-self.history_size:i+self.future_size, :], dest_sel=np.s_[:, :])

                if __mask.all():
                    self.__valid_indices.append(i-self.history_size)

    def __select_random_indices(self):

        self.train_indices = random.sample(
            self.__valid_indices, k=self.__split[0]*len(self.__valid_indices)//100)
        __idxs = set(self.__valid_indices)
        __t_idxs = set(self.train_indices)

        self.dev_indices = random.sample(
            list(__idxs-__t_idxs), k=self.__split[1]*len(self.__valid_indices)//100)

        self.test_indices = list(__idxs - __t_idxs - set(self.dev_indices))

        self.train_indices = np.sort(
            np.array(self.train_indices).reshape(-1, 1), axis=0, kind='mergesort')
        self.dev_indices = np.sort(
            np.array(self.dev_indices).reshape(-1, 1), axis=0, kind='mergesort')
        self.test_indices = np.sort(
            np.array(self.test_indices).reshape(-1, 1), axis=0, kind='mergesort')

    def __sequence_generator(self,indices):
        #normalize features
        def z_score(X,_):
            """
            x_prime = (x - mean)/variance
            """
            #select the volume class columns
            X[:,:3] = np.divide(np.subtract(X[:,:3],self.mean),self.variance)

            return X

        # create targets
        def max_volume(Y,_):
            """
            max_volme = class_1 + class_2 + class_3
            """

            Y = np.sum(Y[:,:3],axis=1,keepdims=True)

            #normalize targets
            Y = np.divide(np.subtract(Y,np.sum(self.mean)),np.sum(self.variance))

            return Y

        __window = np.zeros((self.history_size+self.future_size,self.__data_shape[-1]))

        with self.__raw_data as data:
            for i in indices.reshape(-1):
                data["links/this"].read_direct(__window,source_sel=np.s_[i:i+__window.shape[0]])

                X = __window[:self.history_size]

                #selecting class columns
                Y = __window[-1*self.future_size:,:3]

                #target create function
                Y = np.apply_over_axes(max_volume,Y,[1])
                #normalize feature function
                X = np.apply_over_axes(z_score,X,[1])

                yield X,Y.reshape(-1)

    def __load_raw_data(self, dataset, dtype, raw_row_idx=None, raw_col_idx=None, chunksize=1000):

        with self.__raw_data as data:
            __arr = np.zeros((chunksize, data[dataset].shape[-1]), dtype=dtype)

            if raw_row_idx is not None:
                __lo = raw_row_idx.shape[0] - \
                    (raw_row_idx.shape[0]//chunksize)*chunksize
                __t = more_itertools.grouper(
                    chunksize, raw_row_idx[:-__lo].reshape(-1))
            else:
                __lo = self.__data_shape[0] - \
                    (self.__data_shape[0]//chunksize)*chunksize
                __t = more_itertools.grouper(
                    chunksize, range(0, self.__data_shape[0] - __lo))

            for i in __t:
                data[dataset].read_direct(__arr, source_sel=np.s_[
                                          i])

                if raw_col_idx is not None:
                    yield __arr[:, raw_col_idx]
                else:
                    yield __arr

            # left over chunk
            if raw_row_idx is not None:
                __t = raw_row_idx[-1*__lo:].reshape(-1)
            else:
                __t = list(
                    range(self.__data_shape[0]-__lo, self.__data_shape[0]))

            __arr = np.zeros((__lo, data[dataset].shape[-1]), dtype=dtype)
            data[dataset].read_direct(__arr, source_sel=np.s_[
                                      __t])

            if raw_col_idx is not None:
                yield __arr[:, raw_col_idx]
            else:
                yield __arr

    def __init_params(self):
        __valid_mask_gen = self.__load_raw_data(
            "links/this", int, raw_row_idx=self.train_indices, raw_col_idx=[0, 1, 2], chunksize=1000)

        __sum = np.zeros((1, 3), dtype=int)
        __pow_sum = np.zeros((1, 3), dtype=int)

        for data in __valid_mask_gen:
            __sum = np.add(__sum, np.sum(data, axis=0, keepdims=True))
            __pow_sum = np.add(__pow_sum, np.sum(
                np.power(data, 2), axis=0, keepdims=True))

        self.mean = np.divide(__sum, len(self.train_indices))
        self.variance = np.sqrt(
            __pow_sum/len(self.train_indices) - np.power(self.mean, 2))

    def __store_params(self):
        def write(f):
            f.create_group("params")
            f["params"].create_dataset(
                'train', data=self.train_indices, dtype=int)
            f["params"].create_dataset('dev', data=self.dev_indices, dtype=int)
            f["params"].create_dataset(
                'test', data=self.test_indices, dtype=int)
            f["params"].create_dataset('normalize', data=np.concatenate(
                [self.mean, self.variance], axis=0))
            f["params"].create_dataset("hist_fut", data=np.array(
                [self.history_size, self.future_size], dtype=int))

            print("stored")

        with self.__raw_data as f:
            try:
                write(f)
            except:
                print("deleting existing dataset")
                del f["params"]
                write(f)

    def __load_params(self):
        with self.__raw_data as f:
            __norm_arr = np.zeros(f["params/normalize"].shape)
            __hist_fut_arr = np.zeros(f["params/hist_fut"].shape,dtype=int)

            self.train_indices = np.zeros(
                f["params/train"].shape, dtype=int)
            self.dev_indices = np.zeros(f["params/dev"].shape, dtype=int)
            self.test_indices = np.zeros(f["params/test"].shape, dtype=int)

            f["params/train"].read_direct(self.train_indices)
            f["params/dev"].read_direct(self.dev_indices)
            f["params/test"].read_direct(self.test_indices)
            f["params/normalize"].read_direct(__norm_arr)
            f["params/hist_fut"].read_direct(__hist_fut_arr)

            self.history_size = __hist_fut_arr[0]
            self.future_size = __hist_fut_arr[1]
            self.mean = __norm_arr[0].reshape(1, -1)
            self.variance = __norm_arr[1].reshape(1, -1)


if __name__ == "__main__":
    tdt = train_dev_test("./Data/single_vds.hdf5")
    tdt.init(load_params=True)
    
    
    # for i,(X,Y) in enumerate(tdt.test_ds()):
    #     print(i,X.shape,Y.shape)
    #     if i == 10:
    #         break





