set -e
chunkDir='/home/mohamed/PhD/datasets/yelp_dataset/final_reviews/*'
outDir='/home/mohamed/PhD/datasets/yelp_dataset/bank'
counter=0
for chunk in $chunkDir; do
    python ./format_reviews_to_nn.py --review_data $chunk --output $outDir'/bank_chunk'$counter'.txt'
    counter=$((counter+1));
done