#python run.py --gat_bert --embedding_type bert --output_dir data/output-gcn --dropout 0.2 --hidden_size 200 --learning_rate 5e-5 --exp_name default --per_gpu_train_batch_size 16 --use_ner_feature --use_cross_attn

# 下面是最好的参数设定，在restaurant数据集上，acc：86.88
python run.py --gat_bert \
  --embedding_type bert \
  --output_dir data/output-gcn \
  --dropout 0.1 \
  --hidden_size 300 \
  --learning_rate 5e-5 \
  --exp_name default \
  --per_gpu_train_batch_size 64 \
  --use_bert_global t \
  --use_gat_feature f \
  --use_ner_feature f \
  --use_cross_attn f \
  --cross_attn_heads 4 \
  --alpha 0.2 \
  --save_model t
#
##R-GAT+BERT in laptop
#python run.py --gat_bert \
#  --embedding_type bert \
#  --dataset_name laptop \
#  --output_dir data/output-gcn-laptop \
#  --dropout 0.3 \
#  --num_heads 7 \
#  --hidden_size 200 \
#  --learning_rate 5e-5 \
#  --exp_name default \
#  --per_gpu_train_batch_size 16 \
#  --use_bert_global t \
#  --use_gat_feature f \
#  --use_ner_feature f \
#  --use_cross_attn f \
#  --cross_attn_heads 4 \
#  --alpha 0.2 \
#  --save_model t
###
####R-GAT+BERT in twitter
#python run.py --gat_bert \
#  --embedding_type bert \
#  --dataset_name twitter \
#  --output_dir data/output-gcn-twitter \
#  --dropout 0.2  \
#  --hidden_size 200 \
#  --learning_rate 5e-5 \
#  --exp_name default \
#  --per_gpu_train_batch_size 16 \
#  --use_bert_global t \
#  --use_gat_feature f \
#  --use_ner_feature f \
#  --use_cross_attn f \
#  --cross_attn_heads 4 \
#  --alpha 0.2 \
#  --save_model t \
#  --num_train_epochs 10


#python run.py --gat_our --highway --num_heads 7 --dropout 0.8 # R-GAT in restaurant
#python run.py --gat_our --dataset_name laptop --output_dir data/output-gcn-laptop --highway --num_heads 9 --per_gpu_train_batch_size 32 --dropout 0.7 --num_layers 3 --hidden_size 400 --final_hidden_size 400 # R-GAT in laptop
#python run.py --gat_our --dataset_name twitter --output_dir data/output-gcn-twitter --highway --num_heads 9 --per_gpu_train_batch_size 8 --dropout 0.6 --num_mlps 1 --final_hidden_size 400 # R-GAT in laptop