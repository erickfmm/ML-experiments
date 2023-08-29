cd data/train_data/Audio
echo $(pwd)
sh ./download.sh
cd ../../..

cd ./data/train_data/Emotions_Images
/bin/sh ./download.sh
cd ../../..

cd ./data/train_data/Images_Supervised
/bin/sh ./download.sh
cd ../../..

cd ./data/train_data/Images_Unsupervised
/bin/sh ./download.sh
cd ../../..

cd ./data/train_data/NLP_ENG
/bin/sh ./download.sh
cd ../../..

cd ./data/train_data/NLP_ENG_Sentiment
/bin/sh ./download.sh
cd ../../..

cd ./data/train_data/NLP_ESP_Translation
/bin/sh ./download.sh
cd ../../..

cd ./data/train_data/Emotions_EEG
/bin/sh ./download.sh
cd ../../..

cd ./data/train_data/Emotions_Voice
/bin/sh ./download.sh
cd ../../..

cd ./data/train_data/Manga_Anime
/bin/sh ./download.sh
cd ../../..

cd ./data/train_data/NLP_ENG_Dialogs
/bin/sh ./download.sh
cd ../../..

cd ./data/train_data/NLP_ESP
/bin/sh ./download.sh
cd ../../..

cd ./data/train_data/NLP_Multilingual
/bin/sh ./download.sh
cd ../../..
