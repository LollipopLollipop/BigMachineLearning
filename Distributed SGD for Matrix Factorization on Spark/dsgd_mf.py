import collections
import os
import sys
import numpy as np
#from time import time
from numpy.random import rand
from pyspark import SparkContext
import time
import pyspark
import csv
MIN_EPS = 0.005
TINY_EPS = 0.00001
#MAX_WORD_ID = -1
#MAX_DOC_ID = -1


def map_line(line):
	# TODO: map the line to a triple (word id, doc id, tfidf)
	#global MAX_WORD_ID
	#global MAX_DOC_ID
	elements = line.split(",")
	word_id = int(elements[0])
	doc_id = int(elements[1])
	tfidf = float(elements[2])
	#MAX_WORD_ID = max(MAX_WORD_ID, word_id)
	#MAX_DOC_ID = max(MAX_DOC_ID, doc_id)
	return word_id, doc_id, tfidf

#def save_csv(target_mat,file_path):
#    writer = csv.writer(open(file_path, 'w'))
#    f = open(file_path,'w')
#    for x in xrange(target_mat.shape[0]):
#        s=','.join(['%0.5f' % num for num in target_mat[x,:]])
#        f.write(s+"\n")
#    f.close()


def calculate_loss(pred_matrix, true_matrix):
    # TODO: calculate the loss and RMSE here, and print to stdout
    # Note: pred_matrix is a matrix of num_words x num_docs
    # Note: true_matrix is the result of ratings.collect()
	#score_matrix = np.zeros(pred_matrix.shape)
	#for x,y in true_matrix:
	#	print x+"|"+y
	#	score_matrix[y[0]][y[1]] = y[2]
	nonzero_indice = np.nonzero(true_matrix)
	error = np.sum((pred_matrix[nonzero_indice]-true_matrix[nonzero_indice])**2)
	num_nonzero_entries = np.count_nonzero(true_matrix)

    # TODO: calculate RMSE from error and num_nonzero_entries
	rmse = np.sqrt(error/num_nonzero_entries)
	print('loss: %f, RMSE: %f' % (error, rmse))


def get_worker_id_for_position(word_id, doc_id):
    # TODO: code this. You should be able to determine the worker id from the word id or doc id,
    # or a combination of both. You might also need strata_row_size and/or strata_col_size
    # Suggestion is to return either a column block index or a row block index
    # Assign columns or rows to specific workers
	srs = strata_row_size_bc.value
        #scs = strata_col_size_bc.value
	row_block_idx = word_id / srs
	#col_block_idx = doc_id / strata_col_size
	return row_block_idx


def blockify_matrix(worker_id, partition):
	blocks = collections.defaultdict(list)
	#srs = strata_row_size_bc.value
    	scs = strata_col_size_bc.value
    # Note: partition is a list of (worker_id, (word_id, doc_id, tf_idf))
    # This function should aggregate all elements which belong to the same block
    # Basically, it should return a single tuple for each block in the partition.
    # Each of these tuples should have the format:
    # ( (col_block_index, row_block_index), list of (word_id, doc_id, tf_df))

    # You can use the col_block_index and row_block_index of each of these tuples
    # to determine which of the blocks should be processed on each iteration.

    # Output the blocks. Output should be several of
    # ( (row block index, col block index), list of (word_id, doc_id, tf_idf) )
    # note that worker id is probably one of row block index or col block index
	for wid, data in partition:
		doc_id = data[1]
		col_idx = doc_id / scs    	
		blocks[col_idx,wid].append(data)

	for item in blocks.items():
		yield item


def filter_block_for_iteration(num_iteration, col_block_index, row_block_index, num_workers):
    # TODO: implement me! You might also need the number of workers here
    cur_strata = num_iteration % num_workers
    cur_idx = (col_block_index-row_block_index) %num_workers
    #print ("col:%d row:%d curidx:%d curstrata:%d", col_block_index, row_block_index, cur_idx, cur_strata)
    if (cur_idx == cur_strata):
        return True
    else:
        return False



def perform_sgd(block):
    (col_block, row_block), tuples = block
    #print ("col:%d row:%d tuples:%d", col_block, row_block, len(tuples))
    srs = strata_row_size_bc.value
    scs = strata_col_size_bc.value

    # TODO: determine row_start and col_start based on row_block, col_block and
    # strata_row_size and strata_col_size.
    row_start = srs*row_block # use col_block and row_block to transform to real row and col indexes
    col_start = scs*col_block # use col_block and row_block to transform to real row and col indexes
    w_mat_block = w_mat_bc.value[row_start:row_start + srs, :]
    h_mat_block = h_mat_bc.value[col_start:col_start + scs, :]
    # Note: you need to use block indexes for w_mat_block and h_mat_block to access w_mat_block and h_mat_block

    num_updated = 0
    for word_id, doc_id, tfidf_score in tuples:
        num_updated += 1
        # TODO: update w_mat_block, h_mat_block
        # TODO: update num_updated

        # Note: you might need strata_row_size and strata_col_size to
        # map real word/doc ids to block matrix indexes so you can update
        # w_mat_block and h_mat_block

        # Note: Use MIN_EPS as the min value for the learning rate, in case
        # your learning rate is too small

        # Note: You can use num_old_updates here
        tmp_word_id = word_id - row_start
        tmp_doc_id = doc_id - col_start
        #print ("wid:%d tmpwid:%d did:%d tmpdid:%d", word_id, tmp_word_id, doc_id, tmp_doc_id)
	eps = np.maximum(np.power((100+num_old_updates+num_updated),-beta_value),MIN_EPS)
        #eps = np.power((100+num_old_updates+num_updated),-beta_value)
	#print "eps"+str(eps)
	#print "num_old_updates"+str(num_old_updates)
	w_mat_i = w_mat_block[tmp_word_id,:]
        h_mat_j = h_mat_block[tmp_doc_id,:]
        cur_product = np.dot(w_mat_i, h_mat_j.transpose())
        tmp_w_mat_i = w_mat_i + np.dot(eps*2*(tfidf_score-cur_product),h_mat_j)
        tmp_h_mat_j = h_mat_j + np.dot(eps*2*(tfidf_score-cur_product),w_mat_i)
        w_mat_block[tmp_word_id,:] = tmp_w_mat_i
        h_mat_block[tmp_doc_id,:] = tmp_h_mat_j
	#print ("%0.5f %0.5f %0.5f", tfidf_score, cur_product, np.dot(tmp_w_mat_i, tmp_h_mat_j.transpose()))

    return row_block, col_block, w_mat_block, h_mat_block, num_updated


if __name__ == '__main__':
    # read command line arguments
    num_factors = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    beta_value = float(sys.argv[4])
    inputV_filepath, outputW_filepath, outputH_filepath = sys.argv[5:]

    # create spark context
    conf = pyspark.SparkConf().setAppName("SGD").setMaster("local[{0}]".format(num_workers))
    sc = pyspark.SparkContext(conf=conf)

    # TODO: measure time starting here
    start_time = time.time()

    # get tfidf_scores RDD from data
    # Note: you need to implement the function 'map_line' above.
    if os.path.isfile(inputV_filepath):
        # local file
        tfidf_scores = sc.textFile(inputV_filepath).map(map_line)
    else:
        # directory, or on HDFS
        rating_files = sc.wholeTextFiles(inputV_filepath)
        tfidf_scores = rating_files.flatMap(
            lambda pair: map_line(pair[1]))


    # TODO: get the max_word_id and max_doc_id.
    # this can be coded in 1-2 lines, or 10-20 lines, depending on your approach...
    max_word_id = tfidf_scores.map(lambda a: a[0]).reduce(lambda a,b : a if int(a) > int(b) else b)
    max_doc_id = tfidf_scores.map(lambda a: a[1]).reduce(lambda a,b : a if int(a) > int(b) else b)

    #print "maxword"+str(max_word_id)
    #print "size"+str(len(tfidf_scores.collect()))
    # build W and H as numpy matrices, initialized randomly with ]0,1] values
    w_mat = rand(max_word_id + 1, num_factors) + TINY_EPS
    h_mat = rand(max_doc_id + 1, num_factors) + TINY_EPS
    w_mat = w_mat.astype(np.float32, copy=False)
    h_mat = h_mat.astype(np.float32, copy=False)
    score_matrix = np.zeros([max_word_id + 1,max_doc_id + 1])
    for x,y,z in tfidf_scores.collect():
        score_matrix[x][y] = z

    # TODO: determine strata block size.
    strata_col_size = -(-(max_doc_id+1) // num_workers)
    strata_row_size = -(-(max_word_id+1) // num_workers)
    #print "strata_col_size"+str(strata_col_size)
    strata_col_size_bc = sc.broadcast(strata_col_size)
    strata_row_size_bc = sc.broadcast(strata_row_size)

    # TODO: map scores to (worker id, (word id, doc id, tf idf score) (implement get_worker_id_for_position)
    # Here we are assigning each cell of the matrix to a worker
    tfidf_scores = tfidf_scores.map(lambda score: (
        get_worker_id_for_position(score[0], score[1]),
        (
            score[0], # word id
            score[1], # doc id
            score[2]  # tf idf score
        )
        # partitionBy num_workers, by doing this we are distributing the
        # partitions of the RDD to all of the workers. Each worker gets one partition.
        # Lastly, we do a mapPartitionsWithIndex so each worker can group together
        # all cells that belong to the same block.
        # Make sure we preserve partitioning for correctness and parallelism efficiency
    )).partitionBy(num_workers) \
      .mapPartitionsWithIndex(blockify_matrix, preservesPartitioning=True) \
      .cache()

    #for s,t in tfidf_scores.collect():
    #    print ("col:%d row:%d wid:%d did:%d tfidf:%0.5f",s[0], s[1], t[0], t[1], t[2])

    # finally, run sgd. Each iteration should update one strata.
    num_old_updates = 0
    for current_iteration in range(num_iterations):
        #print "current_iteration:"+str(current_iteration)
	# perform updates for one strata in parallel

        # broadcast factor matrices to workers
        w_mat_bc = sc.broadcast(w_mat)
        h_mat_bc = sc.broadcast(h_mat)

        # perform_sgd should return a tuple consisting of:
        # (row block index, col block index, updated w block, updated h block, number of updates done)
        # s[0][0] is the col or row block index, s[0][1] is the col or row block index
        updated = tfidf_scores \
            .filter(lambda s: filter_block_for_iteration(current_iteration, s[0][0], s[0][1], num_workers)) \
            .map(perform_sgd, preservesPartitioning=True) \
            .collect()

        # unpersist outdated old factor matrices
        w_mat_bc.unpersist()
        h_mat_bc.unpersist()
        # aggregate the updates, update the local w_mat and h_mat
        for block_row, block_col, updated_w, updated_h, num_updates in updated:
            # TODO: update w_mat and h_mat matrices
	    #print ("block_row: %d, block_col: %d", block_row, block_col)
            # map block_row block_col to real indexes (words and doc ids)
            w_update_start = block_row*strata_row_size
            w_update_end = (block_row+1)*strata_row_size
            w_mat[w_update_start:w_update_end, :] = updated_w

            h_update_start = block_col*strata_col_size
            h_update_end = (block_col+1)*strata_col_size
            h_mat[h_update_start:h_update_end, :] = updated_h

            num_old_updates += num_updates

        # TODO: you may want to call calculate_loss here for your experiments
        #calculate_loss(np.dot(w_mat, h_mat.transpose()), score_matrix)

    # TODO: print running time
    print ("running time in %0.3fs" % (time.time()-start_time))
    #print "size"+str(len(tfidf_scores.collect()))
    calculate_loss(np.dot(w_mat, h_mat.transpose()), score_matrix)
    # Stop spark
    sc.stop()

    # TODO: print w_mat and h_mat to outputW_filepath and outputH_filepath
    #save_csv(w_mat,outputW_filepath)
    #save_csv(h_mat.transpose(),outputH_filepath)
    np.savetxt(outputW_filepath, w_mat, fmt='%0.5f', delimiter=",")
    np.savetxt(outputH_filepath, h_mat.transpose(), fmt='%0.5f', delimiter=",")
