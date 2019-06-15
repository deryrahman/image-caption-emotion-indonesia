import os
from dotenv import load_dotenv
load_dotenv()

# configuration
DEBUG = os.getenv('DEBUG') == 'true'
BACKEND_HOST = os.getenv('BACKEND_HOST')
BACKEND_HOST_PORT = os.getenv('BACKEND_HOST_PORT')
FRONTEND_HOST = os.getenv('FRONTEND_HOST')
FRONTEND_HOST_PORT = os.getenv('FRONTEND_HOST_PORT')
IMAGE_FOLDER = os.getenv('IMAGE_FOLDER')
VOCAB_PATH = os.getenv('VOCAB_PATH')
CHECKPOINT_PATHS = {
    'nic': {
        'factual': os.getenv('CHECKPOINT_PATH_NIC_FAC'),
        'happy': os.getenv('CHECKPOINT_PATH_NIC_HAP'),
        'sad': os.getenv('CHECKPOINT_PATH_NIC_SAD'),
        'angry': os.getenv('CHECKPOINT_PATH_NIC_ANG')
    },
    'nic_att': {
        'factual': os.getenv('CHECKPOINT_PATH_NIC_ATT_FAC'),
        'happy': os.getenv('CHECKPOINT_PATH_NIC_ATT_HAP'),
        'sad': os.getenv('CHECKPOINT_PATH_NIC_ATT_SAD'),
        'angry': os.getenv('CHECKPOINT_PATH_NIC_ATT_ANG')
    },
    'stylenet': {
        'factual': os.getenv('CHECKPOINT_PATH_STYLENET_FAC'),
        'happy': os.getenv('CHECKPOINT_PATH_STYLENET_HAP'),
        'sad': os.getenv('CHECKPOINT_PATH_STYLENET_SAD'),
        'angry': os.getenv('CHECKPOINT_PATH_STYLENET_ANG')
    },
    'stylenet_att': {
        'factual': os.getenv('CHECKPOINT_PATH_STYLENET_ATT_FAC'),
        'happy': os.getenv('CHECKPOINT_PATH_STYLENET_ATT_HAP'),
        'sad': os.getenv('CHECKPOINT_PATH_STYLENET_ATT_SAD'),
        'angry': os.getenv('CHECKPOINT_PATH_STYLENET_ATT_ANG')
    }
}