from preparation import invoke_emotion_to_dataset, invoke_edited_to_dataset

mongo_dump_path = './dataset/dump/040719.json'
flickr_folder = './dataset/flickr10k'

invoke_edited_to_dataset(mongo_dump_path, flickr_folder)
for mode in ['happy', 'sad', 'angry']:
    invoke_emotion_to_dataset(mongo_dump_path, flickr_folder, 'flickr', mode)
