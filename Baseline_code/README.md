1. (Optionally) Download images from "http://images.cocodataset.org/zips/train2014.zip" and "http://images.cocodataset.org/zips/val2014.zip"
2. For simplicity, you can directly download the image features extraced by FasterRCNN from "https://storage.googleapis.com/up-down-attention/trainval_36.zip"
3. Process the image features by "python3 tsv2feature.py"
4. Process the text data by "python3 preprocess_text.py". In fact, we have executed this instruction and put the relevant output files in the "cache" folder.
5. Train your/baseline models by "python3 main_LXM.py" and the model which performs best on val dataset will be saved in the "saved_models" folder.
6. Test your/baseline models by "python3 test_LXM.py" and the JSON file of prediction is saved in the "saved_models" folder.
7. Finally, "python3 compute_scores.py" can compute the final scores of the JSON file of your predictions.                                                                                                    
