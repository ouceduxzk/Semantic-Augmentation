# Semantic-Augmentation
Evaluation of Semantic Augmentation method

* save each tag to alltags similarity 
* get top k similar tags for each tag based on gradient cutoff
* then update the Obj_tag matrix 



# Steps to run the code
* 1. process_tfidf_wiki.ipynb generate the sparse matrix for all wikipedia keywork and saved
them into many chunks.
* 2. python parse.py 
* 3. do not need to run the calculate_sim.py (I generated them on server, takes some time), which will generate files in query_pkl, where each file corresponding
to a query and its topn similar words. Then it call the cal_tag_tag_sim from util.py to 
calculate tag tag similarity based p percent quantile
* 4. run semantic_augmentation.py 