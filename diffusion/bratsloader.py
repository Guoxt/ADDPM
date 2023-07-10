import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel


class BRATSDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        '''
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_001_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair']
        else:
            self.seqtypes = ['t1', 't1ce', 't2', 'flair', 'seg']

        self.seqtypes_set = set(self.seqtypes)
        self.database = []
        for root, dirs, files in os.walk(self.directory):
            # if there are no subdirs, we have data
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[3]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path=filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out = torch.stack(out)
        if self.test_flag:
            image=out
            image = image[..., 0:128, 20:212,0]     #crop to a size of (224, 224)
            print(image.shape, path)
            return (image, path)
        else:

            image = out[:-1, ...]
            label = out[-1, ...][None, ...]
            # random crop
            ss = 64
            sss = 96
            index_x = np.random.randint(ss,image.shape[1]-ss,size=1)
            index_y = np.random.randint(sss,image.shape[2]-sss,size=1)
            #
            image = image[..., index_x[0]-ss:index_x[0]+ss,index_y[0]-sss:index_y[0]+sss, 0]      #crop to a size of (224, 224)
            label = label[..., index_x[0]-ss:index_x[0]+ss,index_y[0]-sss:index_y[0]+sss, 0]
            #print(image.shape, label.shape)
            return (image, label)

    def __len__(self):
        return len(self.database)
