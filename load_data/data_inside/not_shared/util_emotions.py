from enum import Enum

__all__ = ["DiscreteEmotion", "QuadrantEmotion"]

class DiscreteEmotion(Enum):
    Neutral = 1 #calm
    Sad = 2 #sadness
    Angry = 3 #anger
    Fear = 4 #anxiety, fearful
    Happy = 5 #happiness
    Surprise = 6
    Amusement = 7
    Disgust = 8
    Boredom = 9
    Anxiety = 10 #only for gsr data
    Calm = 11 #only for ravdess, it has both neutral and calm

class QuadrantEmotion(Enum):
    HAHV = 1
    HALV = 2
    LAHV = 3
    LALV = 4
    Neutral = 5