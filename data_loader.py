import h5py
import os

class hdf5_manager:

    def __init__(self,filename,load=False):

        if not load:
            self.__root = h5py.File(f"{filename}.hdf5",'x')
            self.__filename = f"{filename}.hdf5"
            self.__init()

        else:
            assert os.path.isfile(filename)
            self.__filename = filename
            self.__root = h5py.File(filename,'r+')

    def __init(self):
            self.__root.create_group("links")
            self.__root.create_group("datetime")
            self.__root.create_group("masks")

            self.__maxshapes = {
                "datetime/datetime":(None,1),
                "links/this":(None,42),
                "masks/missing_this":(None,1)
                }

    def __init_dataset(self,path,data):
        __path = path.split('/')
        self.__root[__path[0]].create_dataset(
            __path[1], data=data, maxshape=self.__maxshapes[path])

    def close(self):
        self.__root.close()

    def __flush(self):
        self.__root.flush()

    def write(self,path,data):
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

    def read(self,path):
        """
        path -> the path of the dataset including parent -> links/this
        """
        return self.__root[path]

    def __enter__(self):
        try:
            self.__root.mode
            return self.__root
        except ValueError:
            self.__root = h5py.File(self.__filename,'r+')
            return self.__root

    def __exit__(self, *args, **kwargs):
        self.close()
        




