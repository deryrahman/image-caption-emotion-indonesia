from preparation import process_caption_index_helper, invoke_emotion_to_dataset, invoke_edited_to_dataset

mongo_dump_path = './dataset/dump/040719.json'
path = './dataset/caption_bc.json'
flickr_folder = './dataset/flickr10k'
coco_folder = './dataset/mscoco'
save_path = './dataset/cache/caption_index.helper'
image_id_to_idx = process_caption_index_helper(path, flickr_folder, coco_folder,
                                               save_path)
for mode in ['happy', 'sad', 'angry']:
    invoke_emotion_to_dataset(mongo_dump_path, image_id_to_idx, flickr_folder,
                              'flickr', mode)
invoke_edited_to_dataset(mongo_dump_path, flickr_folder)
