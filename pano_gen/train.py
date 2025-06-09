import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from dataset import PureASGTrainLDMHDR

if __name__ == '__main__':
    resume_path = './models/control_sd15_clip_hdr_share.ckpt'

    batch_size = 4
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    only_mid_control = False

    model = create_model('./models/cldm_v15_clip_hdr_share.yaml').cpu()

    model.load_state_dict(load_state_dict(resume_path, location='cpu'))

    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    data_root = '/mnt/data1/ssy/ldr-inpainting/dataset'
    dataset = PureASGTrainLDMHDR(size=512,
                                 img_root=f'{data_root}/input-rotate-flip-256',
                                 pano_root=f'{data_root}/pano-rotate-flip-512',
                                 asg_root=f'{data_root}/asg-rotate-flip-512',
                                 sg_root=f'{data_root}/sg-rotate-flip-512', )
    dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True, persistent_workers=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    model_ckpt = ModelCheckpoint(every_n_epochs=1, save_top_k=-1, enable_version_counter=False)

    trainer = pl.Trainer(
        devices=1,
        precision='16-mixed',
        strategy='ddp_find_unused_parameters_true',
        max_epochs=10,
        callbacks=[model_ckpt, logger]
    )

    trainer.fit(model, dataloader)
