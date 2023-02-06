import bot
from sc2 import run_game, maps, Race, Difficulty
from sc2.player import Bot, Computer
import keras  # Keras 2.1.2 and TF-GPU 1.8.0
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import os


if __name__ == '__main__':
    for i in range(10):
        run_game(maps.get("AcropolisLE"), [
            Bot(Race.Protoss, bot.DiavolosBot(use_model=False)),
            # Bot(Race.Protoss, bot.DiavolosBot(use_model=False, title=2))
            Computer(Race.Random, Difficulty.Medium)
        ], realtime=False)
