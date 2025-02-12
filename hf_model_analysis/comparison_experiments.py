from transformers import AutoModelForObjectDetection, AutoModel, \
    SegformerForSemanticSegmentation, AutoModelForImageClassification, AutoModelForImageTextToText

from hf_model_analysis.comparison_run import ModelComparisonRun

HF_CACHE_PATH = "/mount-fs/hf-home"
SSD_MODEL_CMP_OUT = "/mount-ssd/model-cmp/out"

#####################################
# Object Detection
#####################################
object_detection_comparison_runs = []

detr_resnet_50_cmp = ModelComparisonRun(
    base_model_id="facebook/detr-resnet-50",
    base_model_class=AutoModelForObjectDetection,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:facebook%2Fdetr-resnet-50&sort=downloads",
    start_page_num=0,
    end_page_num=10,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
object_detection_comparison_runs.append(detr_resnet_50_cmp)

detr_resnet_101_cmp = ModelComparisonRun(
    base_model_id="facebook/detr-resnet-101",
    base_model_class=AutoModelForObjectDetection,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:facebook/detr-resnet-101",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
object_detection_comparison_runs.append(detr_resnet_101_cmp)

conditional_detr_resnet_50 = ModelComparisonRun(
    base_model_id="microsoft/conditional-detr-resnet-50",
    base_model_class=AutoModelForObjectDetection,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft%2Fconditional-detr-resnet-50&sort=downloads",
    start_page_num=0,
    end_page_num=2,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
object_detection_comparison_runs.append(conditional_detr_resnet_50)

deformable_detr_resnet_50 = ModelComparisonRun(
    base_model_id="SenseTime/deformable-detr",
    base_model_class=AutoModelForObjectDetection,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:SenseTime/deformable-detr",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
object_detection_comparison_runs.append(deformable_detr_resnet_50)

detr_resnet_50_dc5 = ModelComparisonRun(
    base_model_id="facebook/detr-resnet-50-dc5",
    base_model_class=AutoModelForObjectDetection,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:facebook/detr-resnet-50-dc5",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
object_detection_comparison_runs.append(detr_resnet_50_dc5)

table_transformer_structure_recognition = ModelComparisonRun(
    base_model_id="microsoft/table-transformer-structure-recognition",
    base_model_class=AutoModelForObjectDetection,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/table-transformer-structure-recognition",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
object_detection_comparison_runs.append(table_transformer_structure_recognition)

yolo_small = ModelComparisonRun(
    base_model_id="hustvl/yolos-small",
    base_model_class=AutoModelForObjectDetection,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:hustvl/yolos-small",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
object_detection_comparison_runs.append(yolo_small)

yolos_tiny = ModelComparisonRun(
    base_model_id="hustvl/yolos-tiny",
    base_model_class=AutoModelForObjectDetection,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:hustvl/yolos-tiny",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
object_detection_comparison_runs.append(yolos_tiny)

table_transformer_detection = ModelComparisonRun(
    base_model_id="microsoft/table-transformer-detection",
    base_model_class=AutoModelForObjectDetection,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/table-transformer-detection",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
object_detection_comparison_runs.append(table_transformer_detection)

#####################################
# Image Feature Extraction
#####################################
feature_ext_comparison_runs = []

vit_base_patch16_224_in21k = ModelComparisonRun(
    base_model_id="google/vit-base-patch16-224-in21k",
    base_model_class=AutoModel,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:google%2Fvit-base-patch16-224-in21k&sort=downloads",
    start_page_num=0,
    end_page_num=10,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
feature_ext_comparison_runs.append(vit_base_patch16_224_in21k)

vit_large_patch16_224_in21k = ModelComparisonRun(
    base_model_id="google/vit-large-patch16-224-in21k",
    base_model_class=AutoModel,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:google%2Fvit-large-patch16-224-in21k&sort=downloads",
    start_page_num=0,
    end_page_num=2,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
feature_ext_comparison_runs.append(vit_large_patch16_224_in21k)

vit_huge_patch16_224_in21k = ModelComparisonRun(
    base_model_id="google/vit-huge-patch14-224-in21k",
    base_model_class=AutoModel,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:google/vit-huge-patch14-224-in21k",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
feature_ext_comparison_runs.append(vit_huge_patch16_224_in21k)

dinov2_base = ModelComparisonRun(
    base_model_id="facebook/dinov2-base",
    base_model_class=AutoModel,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:facebook/dinov2-base",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
feature_ext_comparison_runs.append(dinov2_base)

dinov2_large = ModelComparisonRun(
    base_model_id="facebook/dinov2-large",
    base_model_class=AutoModel,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:facebook/dinov2-large",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
feature_ext_comparison_runs.append(dinov2_large)

#####################################
# Image Segmentation
#####################################
image_segmentation_comparison_runs = []

segformer_b0_finetuned_ade_512_512 = ModelComparisonRun(
    base_model_id="nvidia/segformer-b0-finetuned-ade-512-512",
    base_model_class=SegformerForSemanticSegmentation,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:nvidia/segformer-b0-finetuned-ade-512-512",
    start_page_num=0,
    end_page_num=2,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_segmentation_comparison_runs.append(segformer_b0_finetuned_ade_512_512)

segformer_b1_finetuned_ade_512_512 = ModelComparisonRun(
    base_model_id="nvidia/segformer-b1-finetuned-ade-512-512",
    base_model_class=SegformerForSemanticSegmentation,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:nvidia/segformer-b1-finetuned-ade-512-512",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_segmentation_comparison_runs.append(segformer_b1_finetuned_ade_512_512)

segformer_b2_finetuned_ade_512_512 = ModelComparisonRun(
    base_model_id="nvidia/segformer-b2-finetuned-ade-512-512",
    base_model_class=SegformerForSemanticSegmentation,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:nvidia/segformer-b2-finetuned-ade-512-512",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_segmentation_comparison_runs.append(segformer_b2_finetuned_ade_512_512)

segformer_b1_finetuned_cityscapes_1024_1024 = ModelComparisonRun(
    base_model_id="nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    base_model_class=SegformerForSemanticSegmentation,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_segmentation_comparison_runs.append(segformer_b1_finetuned_cityscapes_1024_1024)

segformer_b5_finetuned_cityscapes_1024_1024 = ModelComparisonRun(
    base_model_id="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    base_model_class=SegformerForSemanticSegmentation,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_segmentation_comparison_runs.append(segformer_b5_finetuned_cityscapes_1024_1024)

#####################################
# Image Classification
#####################################
image_classification_comparison_runs = []

mobilenet_v2_0_224 = ModelComparisonRun(
    base_model_id="google/mobilenet_v2_1.0_224",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:google/mobilenet_v2_1.0_224",
    start_page_num=0,
    end_page_num=2,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(mobilenet_v2_0_224)

efficientnet_b0 = ModelComparisonRun(
    base_model_id="google/efficientnet-b0",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:google/efficientnet-b0",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(efficientnet_b0)

resnet50 = ModelComparisonRun(
    base_model_id="microsoft/resnet-50",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/resnet-50",
    start_page_num=0,
    end_page_num=5,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(resnet50)

resnet_18 = ModelComparisonRun(
    base_model_id="microsoft/resnet-18",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/resnet-18",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(resnet_18)

resnet_152 = ModelComparisonRun(
    base_model_id="microsoft/resnet-152",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/resnet-152",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(resnet_152)

resnet_101 = ModelComparisonRun(
    base_model_id="microsoft/resnet-101",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/resnet-101",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(resnet_101)

vit_tiny_patch16_224 = ModelComparisonRun(
    base_model_id="WinKawaks/vit-tiny-patch16-224",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:WinKawaks/vit-tiny-patch16-224",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(vit_tiny_patch16_224)

vit_small_patch16_224 = ModelComparisonRun(
    base_model_id="WinKawaks/vit-small-patch16-224",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:WinKawaks/vit-small-patch16-224",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(vit_small_patch16_224)

vit_base_patch16_224 = ModelComparisonRun(
    base_model_id="google/vit-base-patch16-224",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:google/vit-base-patch16-224",
    start_page_num=0,
    end_page_num=18,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(vit_base_patch16_224)

beit_base_patch16_224_pt22k_ft22k = ModelComparisonRun(
    base_model_id="microsoft/beit-base-patch16-224-pt22k-ft22k",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/beit-base-patch16-224-pt22k-ft22k",
    start_page_num=0,
    end_page_num=3,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(beit_base_patch16_224_pt22k_ft22k)

mobilevit_small = ModelComparisonRun(
    base_model_id="apple/mobilevit-small",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:apple/mobilevit-small",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(mobilevit_small)

swinv2_tiny_patch4_window16_256 = ModelComparisonRun(
    base_model_id="microsoft/swinv2-tiny-patch4-window16-256",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/swinv2-tiny-patch4-window16-256",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(swinv2_tiny_patch4_window16_256)

vit_large_patch32_384 = ModelComparisonRun(
    base_model_id="google/vit-large-patch32-384",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:google/vit-large-patch32-384",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(vit_large_patch32_384)

mit_b0 = ModelComparisonRun(
    base_model_id="nvidia/mit-b0",
    base_model_class=AutoModelForImageClassification,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:nvidia/mit-b0",
    start_page_num=0,
    end_page_num=12,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_classification_comparison_runs.append(mit_b0)

#####################################
# Image to Text
#####################################
image_to_text_comparison_runs = []

trocr_base_handwritten = ModelComparisonRun(
    base_model_id="microsoft/trocr-base-handwritten",
    base_model_class=AutoModelForImageTextToText,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/trocr-base-handwritten",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_to_text_comparison_runs.append(trocr_base_handwritten)

trocr_large_printed = ModelComparisonRun(
    base_model_id="microsoft/trocr-large-printed",
    base_model_class=AutoModelForImageTextToText,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/trocr-large-printed",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_to_text_comparison_runs.append(trocr_large_printed)

blip_image_captioning_base = ModelComparisonRun(
    base_model_id="Salesforce/blip-image-captioning-base",
    base_model_class=AutoModelForImageTextToText,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:Salesforce/blip-image-captioning-base",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_to_text_comparison_runs.append(blip_image_captioning_base)

donut_base_finetuned_cord_v2 = ModelComparisonRun(
    base_model_id="naver-clova-ix/donut-base-finetuned-cord-v2",
    base_model_class=AutoModelForImageTextToText,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:naver-clova-ix/donut-base-finetuned-cord-v2",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_to_text_comparison_runs.append(donut_base_finetuned_cord_v2)

trocr_base_printed = ModelComparisonRun(
    base_model_id="microsoft/trocr-base-printed",
    base_model_class=AutoModelForImageTextToText,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/trocr-base-printed",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_to_text_comparison_runs.append(trocr_base_printed)

vit_gpt2_image_captioning = ModelComparisonRun(
    base_model_id="nlpconnect/vit-gpt2-image-captioning",
    base_model_class=AutoModelForImageTextToText,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:nlpconnect/vit-gpt2-image-captioning",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_to_text_comparison_runs.append(vit_gpt2_image_captioning)

git_base = ModelComparisonRun(
    base_model_id="microsoft/git-base",
    base_model_class=AutoModelForImageTextToText,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/git-base",
    start_page_num=0,
    end_page_num=4,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_to_text_comparison_runs.append(git_base)

trocr_base_stage1 = ModelComparisonRun(
    base_model_id="microsoft/trocr-base-stage1",
    base_model_class=AutoModelForImageTextToText,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:microsoft/trocr-base-stage1",
    start_page_num=None,
    end_page_num=None,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_to_text_comparison_runs.append(trocr_base_stage1)

donut_base = ModelComparisonRun(
    base_model_id="naver-clova-ix/donut-base",
    base_model_class=AutoModelForImageTextToText,
    base_model_url="https://huggingface.co/models?other=base_model:finetune:naver-clova-ix/donut-base",
    start_page_num=0,
    end_page_num=13,
    result_output_path=SSD_MODEL_CMP_OUT,
    hf_cache_path=HF_CACHE_PATH
)
image_to_text_comparison_runs.append(donut_base)

if __name__ == '__main__':
    comparison_run_categories = \
        [image_to_text_comparison_runs, image_classification_comparison_runs, image_segmentation_comparison_runs,
         feature_ext_comparison_runs, object_detection_comparison_runs]
    for run_category in comparison_run_categories:
        for comparison_run in run_category:
            comparison_run.start_comparison()
