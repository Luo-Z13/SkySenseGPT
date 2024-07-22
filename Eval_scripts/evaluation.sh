

########  Evaluation
model_name='your_model_name'

# FIT-RSFG Scene Classification
python geochat/eval/batch_geochat_sceneclassification.py \
    --model-path /your_model_path/${model_name} \
    --question-file /FIT-RSFG-Bench/test_FITRS_imageclassify_eval.jsonl \
    --answers-file /output_answers/${model_name}/FITRS_imageclassify_eval_${model_name}.jsonl \
    --image-folder /img_path/imgv2_split_512_100_vaild
# FIT-RSFG ImageCaption
python geochat/eval/batch_geochat_caption.py \
    --model-path /your_model_path/${model_name} \
    --question-file /FIT-RSFG-Bench/test_FITRS_image_caption_eval.jsonl \
    --answers-file /output_answers/${model_name}/FITRS_image_caption_answer_${model_name}.jsonl \
    --image-folder /img_path/imgv2_split_512_100_vaild
# FIT-RSFG RegionCaption
python geochat/eval/batch_geochat_caption.py \
    --model-path /your_model_path/${model_name} \
    --question-file /FIT-RSFG-Bench/test_FITRS_region_caption_eval.jsonl \
    --answers-file /output_answers/${model_name}/FITRS_region_caption_answer_${model_name}.jsonl \
    --image-folder /img_path/imgv2_split_512_100_vaild
# FIT-RSFG VQA
python geochat/eval/batch_geochat_vqa.py \
    --model-path /your_model_path/${model_name} \
    --question-file /FIT-RSFG-Bench/test_FITRS_vqa_eval.jsonl \
    --answers-file /output_answers/${model_name}/FITRS_vqa_eval_${model_name}.jsonl \
    --image-folder /img_path/imgv2_split_512_100_vaild
## FIT-RSFG ComplexComprehension
python geochat/eval/batch_geochat_complex_compre.py \
    --model-path /your_model_path/${model_name} \
    --question-file /FIT-RSFG-Bench/test_FITRS_complex_comprehension_eval.jsonl \
    --answers-file /output_answers/${model_name}/FITRS_complex_comprehension_eval_${model_name}.jsonl \
    --image-folder /img_path/imgv2_split_512_100_vaild
# FIT-RSRC Single-Choice
python geochat/eval/batch_fitrsrc_single_choice_qa.py \
    --model-path /your_model_path/${model_name} \
    --question-file /FIT-RSFG-Bench/FIT-RSRC_Questions_2k.jsonl \
    --answers-file /output_answers/${model_name}/FIT-RSRC_singlechoice_eval_${model_name}.jsonl \
    --image-folder /img_path/imgv2_split_512_100_vaild

######## Evaluation ######
# eval FIT-RSFG Caption
python Eval/pycocoevalcap/eval_custom_caption.py \
       --root_path /output_answers/ \
       --model_answers_file_list \
       "geochat-7B/FITRS_image_caption_answer_geochat-7B.jsonl" \
       "geochat-7B/FITRS_region_caption_answer_geochat-7B.jsonl"

# eval FIT-RSFG ComplexComprehension
python Eval/eval_complex_comprehension.py \
       --answer-file /output_answers/${model_name}/FITRS_complex_comprehension_eval_${model_name}.jsonl

######
# VQA-HRBEN
python geochat/eval/batch_geochat_vqa_hrben.py \
    --model-path /your_model_path/${model_name} \
    --question-file /FIT-RSFG-Bench/hrben.jsonl \
    --answers-file /output_answers/${model_name}/HRBEN_answers_fgrs_${model_name}.jsonl \
    --image-folder /HRBEN_RSVQAHR/Data

# eval HRBEN(RSVQA-HR)
python Eval/eval_vqa_HRBEN.py \
    --answer-file /output_answers/${model_name}/HRBEN_answers_fgrs_${model_name}.jsonl \
    --output-file /output_answers/${model_name}/HRBEN_answers_fgrs_${model_name}_combined.jsonl \
    --questions-file Eval/HRBEN/USGS_split_test_phili_answers.json \
    --answers-gt-file Eval/HRBEN/USGS_split_test_phili_answers.json