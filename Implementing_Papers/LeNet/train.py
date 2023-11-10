import torch
import pytorch_lightning as pl
from model import LeLightning
from dataset import MnistDataModule
import config

if __name__ == "__main__":
	# Initialize
	model = LeLightning(num_classes=config.NUM_CLASSES, learning_rate=config.LEARNING_RATE)
	dm = MnistDataModule(config.DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS)
	trainer = pl.Trainer(accelerator=config.ACCELERATOR, 
						max_epochs=config.NUM_EPOCHS,
						enable_checkpointing=False,
						logger=False)

	# Run
	# Something is off here because prepare_data is called every time and the same data is downloaded
	# for train test and val, gotta figure that out
	trainer.fit(model, datamodule=dm)
	trainer.validate(model, datamodule=dm)
	trainer.test(model, datamodule=dm)