import random as rn
import numpy as np
from pathlib import Path
import json
import plot_utils as pt

class Pretraining():
    '''Returns precalculated data from a specified folder
    nepisodes: number of episodes aka folders to open
    dt: desired data sampling from file. must be a multiple of the sampling in the data file
    '''
    def __init__(self, folder, nepisodes, dt):
        self.folder = folder
        self.nepisodes  =nepisodes
        self.dt = dt

        # get list of folders in given root folder
        folders = list(Path(self.folder).iterdir())
        N = len(folders)
        print('Training folder has %i folders'%(N))

        # error if nepisodes exceeds max available
        if (nepisodes > N):
            raise Exception('Pretraining data longer than available')

        # make rnd list
        rn.shuffle(folders)
        folders = folders[0:self.nepisodes]
        self.folders = iter(folders)
        self._check_next_folder()

    def _check_next_folder(self):
        # get next file or folder name
        self.current_folder = next(self.folders, None)
        if self.current_folder is None:
            return False
        # access the data
        init_file = Path(self.current_folder / 'initial-conditions.txt')
        data_file = Path(self.current_folder / 'data')
        phases_ = pt.phase2modpi(np.load(data_file, allow_pickle=True))
        # get data dt
        fp = open(init_file, mode='r')
        initial_cond = json.load(fp)
        dt_save = initial_cond['julia simulation dt save']
        # check whether it matches the desired df, must be multiple
        if (self.dt / dt_save) % 1 != 0:
            raise Exception('Desired dt is not equal or multiple of data file\n'
                            'in %s' % (self.current_folder))
        # calc the right df
        dt_select = self.dt / dt_save
        self.phases = phases_[:,::int(dt_select)]
        self.index = iter(range(self.phases.shape[1]))
        return True

    def get_datapoint(self):
        ep_done = False
        all_done = False
        d = None
        i = next(self.index, None)
        if i == None:
            ep_done = True
            if not self._check_next_folder():
                all_done = True
                print('training data done')
        else:
            d = self.phases[:, i]
        return d, ep_done, all_done


# ----------------- test pretraining
#
# t = Pretraining('/media/blaubaer/data/Language_of_the_brain/Kuramoto_pretraining/Kuramoto_pretraining_data_N30',
#                 5,1)
# for i in range(20000):
#     data, done_ep, done_all = t.get_datapoint()
#     print('data' if data is not None else '   --- None', done_ep, done_all, t.current_folder)
#     if done_all:
#         break
#
