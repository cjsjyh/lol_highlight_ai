__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '3.6'
__status__ = "Research"
__date__ = "1/12/2018"
__license__= "MIT License"

from torch.autograd import Variable
from model.vasnet_model import VASNet
from model.vasnet_audio20_concat import VASNet_Audio20_Concat
from model.vasnet_audio128_concat import VASNet_Audio128_Concat
from model.vasnet_audio20_concat_regressor import VASNet_Audio20_Concat_regressor
from model.vasnet_audio128_att import VASNet_Audio128_Att

class HParameters:

    def __init__(self):
        self.verbose = False
        self.use_cuda = True
        self.cuda_device = 0
        self.max_summary_length = 0.15

        self.l2_req = 0.00001
        self.lr_epochs = [0]
        self.lr = [0.00005]

        self.epochs_max = 300
        self.train_batch_size = 1

        self.output_dir = 'ex-10'

        self.root = ''
        self.model_name = {
                "vasnet":VASNet,
                "vasnet_audio20_concat":VASNet_Audio20_Concat,
                "vasnet_audio128_concat":VASNet_Audio128_Concat,
                "vasnet_audio20":VASNet_Audio20_Concat,
                "vasnet_audio20_regressor":VASNet_Audio20_Concat_regressor,
                "vasnet_audio128_att": VASNet_Audio128_Att
            }
        self.datasets=[#'datasets/a.h5',
                       # 'datasets/b.h5',
                       # 'datasets/c.h5',
                       # 'datasets/d.h5',
                       # 'datasets/e.h5',
                       #'datasets/f.h5',
                       # 'datasets/merge.h5',
                       #'datasets/m2.h5',
                       'datasets/m3_copy.h5'
                       #'datasets/m3_copy.h5'
                       #'datasets/audio.h5'
                       #'datasets/LOL.h5'
                       ]

        self.splits = [# 'splits/a_splits.json',
                       # 'splits/b_splits.json',
                       # 'splits/c_splits.json',
                       # 'splits/d_splits.json',
                       # 'splits/e_splits.json',
                       #'splits/f_splits.json'
                       #'splits/merge_splits.json'
                       #'splits/m2_splits.json',
                       'splits/m3_splits.json'
                       #'splits/audio_splits.json'
                        #'splits/merge_gaussian_splits.json'
                        #'splits/LOL_splits.json'
                        ]

        return


    def get_dataset_by_name(self, dataset_name):
        for d in self.datasets:
            if dataset_name in d:
                return [d]
        return None

    def load_from_args(self, args):
        for key in args:
            val = args[key]
            if val is not None:
                if hasattr(self, key) and isinstance(getattr(self, key), list):
                    val = val.split()
                setattr(self, key, val)
                if key == "model":
                    setattr(self, "model",self.model_name[val])


    def __str__(self):
        vars = [attr for attr in dir(self) if not callable(getattr(self,attr)) and not (attr.startswith("__") or attr.startswith("_"))]

        info_str = ''
        for i, var in enumerate(vars):
            val = getattr(self, var)
            if isinstance(val, Variable):
                val = val.data.cpu().numpy().tolist()[0]
            info_str += '['+str(i)+'] '+var+': '+str(val)+'\n'

        return info_str
    def model(self):
        model_dict = {'vasnet':VASNet()}


if __name__ == "__main__":

    # Tests
    hps = HParameters()
    print(hps)

    args = {'root': 'root_dir',
            'datasets': 'set1,set2,set3',
            'splits': 'split1, split2',
            'new_param_float': 1.23456
            }

    hps.load_from_args(args)
    print(hps)
