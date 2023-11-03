CUDA_VISIBLE_DEVICES=4 python test_video.py \
--content_dir ~/yura/anet_raw/anet_v1-3_merged_256_1fps \
--style_dir styleimages/Leonardo_da_Vinci_102.jpg \
--KC 4 --KS -10 \
--output_dir outputs/Leonardo \
--vgg_path models/vgg_normalised.pth \
--csbnet_path models/csbnet.pth \