import random as rn
import numpy as np
from pathlib import Path
import json
import plot_utils as pt
from collections import deque


class Pretraining():
    '''Returns precalculated data from a specified folder
    nepisodes: number of episodes aka folders to open
    dt: desired data sampling from file. must be a multiple of the sampling in the data file
    '''
    def __init__(self,
                 folder,
                 nepisodes,
                 sniplen,
                 history_len,
                 make_observation_function):
        self.folder = folder
        self.nepisodes  =nepisodes
        self.sniplen = sniplen
        self.history_len = history_len
        self.gym_make_observation = make_observation_function

        self.current_state = deque(maxlen=self.history_len)

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
        self._read_next_folder()

    def _read_next_folder(self):
        # read data from next folder, return False if no folder left
        # read sampling rate from metadata file
        # calculate snip len
        # get next file or folder name
        print(' *** pretraining *** reading next folder')
        self.current_folder = next(self.folders, None)
        if self.current_folder is None:
            return False
        # access the data
        init_file = Path(self.current_folder / 'initial-conditions.txt')
        data_file = Path(self.current_folder / 'data')
        self.phases = pt.phase2modpi(np.load(data_file, allow_pickle=True))
        # get data sampling rate
        fp = open(init_file, mode='r')
        initial_cond = json.load(fp)
        dt_save = initial_cond['julia simulation dt save']
        # check whether it matches the desired df, must be multiple
        if (self.sniplen / dt_save) % 1 != 0:
            raise Exception('Desired sniplen is not equal or multiple of sampling rate\n'
                            'in %s' % (self.current_folder))
        # calc the right df
        self.points_per_snip = self.sniplen / dt_save
        # snips per episode = total episode len divided by snip len
        n_snips_per_episode = int(self.phases.shape[1] // self.points_per_snip)
        # reset state deque
        self.current_state.clear()
        # make a list of snips
        self.sniplist = iter(np.split(self.phases, n_snips_per_episode, axis=1))
        self.current_snip = next(self.sniplist)

        return True

    def get_data_snip(self):
        # return a snip = number of data points which form a window
        ep_done = False
        all_done = False
        data = self.current_snip
        # advance current snip by one
        self.current_snip = next(self.sniplist, None)
        if self.current_snip is None:
            # the current episode is done
            ep_done = True
            print(' *** pretraining episode done ***')
            if not self._read_next_folder():
                # all pretraining episodes are done
                all_done = True
                print(' *** pretraining data done *** ')
        # print(' --- get_data_snip(), d shape, start, stop, current state', data.shape, self.current_state)
        return data, ep_done, all_done

    def _load_single_snip_and_reward(self):
        # get current snip
        snip, done, all_done = self.get_data_snip()
        # call the gym's function to amke the observation
        reward, b,c, center_asyn,e,f = self.gym_make_observation(snip, 0)
        self.current_state.append([center_asyn])
        return reward, done, all_done

    def get_observation_and_reward(self):
        # when starting a new episode, get snips to fill observation history
        while len(self.current_state) < (self.history_len-1):
            a,b,c = self._load_single_snip_and_reward()
        # load new snip
        reward, done, all_done = self._load_single_snip_and_reward()
        # reset episode if done is true

        return reward, np.array([self.current_state]), done, all_done





# # ----------------- test pretraining
#
# t = Pretraining('/media/blaubaer/data/Language_of_the_brain/Kuramoto_pretraining/Kuramoto_pretraining_data_N30',
#                 nepisodes=5,history_len=2, make_observation_function=)
# for i in range(2000):
#     reward, state, done_ep, done_all = t.get_observation_and_reward()
#     print(state.shape if state is not None else '   --- None', done_ep, done_all)
#     if done_all:
#         break

