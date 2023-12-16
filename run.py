# run.py

from train import Trainer

# Configuration parameters
HEIGHT = 256
WIDTH = 256
CHANNELS = 3
EPOCHS = 10
BATCH = 100
CHECKPOINT = 50
TRAIN_PATH_A = r"C:\Users\laure\Dropbox\School\BSE\Coursework\23 Fall\GenerativeAI\code for projects\GAIfinalproject\impressionist_landscapes_resized_1024\train3000"
TRAIN_PATH_B = r"C:\Users\laure\Dropbox\School\BSE\Coursework\23 Fall\GenerativeAI\code for projects\GAIfinalproject\landscape_photos\train2850"

# Initialize Trainer
trainer = Trainer(height=HEIGHT, width=WIDTH, channels=CHANNELS, epochs=EPOCHS,
                  batch=BATCH, checkpoint=CHECKPOINT, 
                  train_data_path_A=TRAIN_PATH_A, train_data_path_B=TRAIN_PATH_B)

# Start Training
trainer.train()
