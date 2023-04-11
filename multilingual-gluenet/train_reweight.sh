script=alignment_mse_multilingual_roberta_2clip_reweight.py

CUDA_VISIBLE_DEVICES=0 python alignment_mse_multilingual_roberta_2clip_reweight.py --DATA_PATH_SRC data/WikiMatrix.en-zh.txt.en --DATA_PATH_TAR data/WikiMatrix.en-zh.txt.zh --DATA_PATH_SRC_1 data_processed/laion-1M-trans-en-zh-cn-en.txt --DATA_PATH_TAR_1 data_processed/laion-1M-trans-en-zh-cn-zh-cn.txt --tarLanguage Chinese


# CUDA_VISIBLE_DEVICES=1 python alignment_mse_multilingual_roberta_2clip_reweight.py --DATA_PATH_SRC data/WikiMatrix.en-ja.txt.en --DATA_PATH_TAR data/WikiMatrix.en-ja.txt.ja --DATA_PATH_SRC_1 data_processed/laion-1M-trans-en-ja-en.txt --DATA_PATH_TAR_1 data_processed/laion-1M-trans-en-ja-ja.txt  --tarLanguage Japanese

# CUDA_VISIBLE_DEVICES=2 python alignment_mse_multilingual_roberta_2clip_reweight.py --DATA_PATH_SRC data/WikiMatrix.en-fr.txt.en --DATA_PATH_TAR data/WikiMatrix.en-fr.txt.fr --DATA_PATH_SRC_1 data_processed/laion-1M-trans-en-fr-en.txt --DATA_PATH_TAR_1 data_processed/laion-1M-trans-en-fr-fr.txt --tarLanguage French

# CUDA_VISIBLE_DEVICES=3 python alignment_mse_multilingual_roberta_2clip_reweight.py --DATA_PATH_SRC data/WikiMatrix.en-it.txt.en --DATA_PATH_TAR data/WikiMatrix.en-it.txt.it --DATA_PATH_SRC_1 data_processed/laion-1M-trans-en-it-en.txt --DATA_PATH_TAR_1 data_processed/laion-1M-trans-en-it-it.txt --tarLanguage Italian

# CUDA_VISIBLE_DEVICES=4 python alignment_mse_multilingual_roberta_2clip_reweight.py --DATA_PATH_SRC data/WikiMatrix.en-es.txt.en --DATA_PATH_TAR data/WikiMatrix.en-es.txt.es --DATA_PATH_SRC_1 data_processed/laion-1M-trans-en-es-en.txt --DATA_PATH_TAR_1 data_processed/laion-1M-trans-en-es-es.txt --tarLanguage Spanish
